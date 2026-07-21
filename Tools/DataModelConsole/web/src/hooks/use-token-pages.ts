"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import type { TokenPage } from "@/types";

export interface UseTokenPagesResult<T> {
  items: T[];
  error: Error | null;
  loading: boolean;
  loadingMore: boolean;
  hasMore: boolean;
  loadMore: () => void;
  reload: () => void;
}

export function useTokenPages<T>(
  fetchPage: (token: string) => Promise<TokenPage<T>>,
  deps: readonly unknown[] = [],
  getKey?: (item: T) => string,
): UseTokenPagesResult<T> {
  const [items, setItems] = useState<T[]>([]);
  const [nextToken, setNextToken] = useState("");
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [generation, setGeneration] = useState(0);
  const coordinateRef = useRef(0);
  const seenTokensRef = useRef(new Set<string>());
  const inFlightTokensRef = useRef(new Set<string>());
  const fetchPageRef = useRef(fetchPage);
  const getKeyRef = useRef(getKey);
  fetchPageRef.current = fetchPage;
  getKeyRef.current = getKey;

  const uniqueItems = useCallback((current: T[], incoming: T[]): T[] => {
    const keyFor = getKeyRef.current;
    if (!keyFor) return [...current, ...incoming];

    const seen = new Set(current.map(keyFor));
    const appended = [...current];
    for (const item of incoming) {
      const key = keyFor(item);
      if (!seen.has(key)) {
        seen.add(key);
        appended.push(item);
      }
    }
    return appended;
  }, []);

  useEffect(() => {
    const coordinate = ++coordinateRef.current;
    seenTokensRef.current = new Set([""]);
    inFlightTokensRef.current = new Set([""]);
    setItems([]);
    setNextToken("");
    setError(null);
    setLoading(true);
    setLoadingMore(false);
    fetchPageRef
      .current("")
      .then((page) => {
        if (coordinate !== coordinateRef.current) return;
        const next = page.next_page_token ?? "";
        if (next && seenTokensRef.current.has(next)) {
          setError(new Error("Upstream pagination token entered a cycle."));
          setNextToken("");
        } else {
          setItems((current) =>
            uniqueItems(current, page.items ?? []),
          );
          setNextToken(next);
        }
        setLoading(false);
      })
      .catch((cause: unknown) => {
        if (coordinate !== coordinateRef.current) return;
        setError(cause instanceof Error ? cause : new Error(String(cause)));
        setLoading(false);
      })
      .finally(() => {
        if (coordinate === coordinateRef.current) {
          inFlightTokensRef.current.delete("");
        }
      });
    return () => {
      if (coordinate === coordinateRef.current) {
        coordinateRef.current += 1;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, generation, uniqueItems]);

  const loadMore = useCallback(() => {
    if (!nextToken || loading || loadingMore) return;
    const coordinate = coordinateRef.current;
    const requestedToken = nextToken;
    if (
      seenTokensRef.current.has(requestedToken) ||
      inFlightTokensRef.current.has(requestedToken)
    ) {
      return;
    }
    seenTokensRef.current.add(requestedToken);
    inFlightTokensRef.current.add(requestedToken);
    setLoadingMore(true);
    setError(null);
    fetchPageRef
      .current(requestedToken)
      .then((page) => {
        if (coordinate !== coordinateRef.current) return;
        const next = page.next_page_token ?? "";
        if (next && seenTokensRef.current.has(next)) {
          setError(new Error("Upstream pagination token entered a cycle."));
          setNextToken("");
        } else {
          setItems((current) =>
            uniqueItems(current, page.items ?? []),
          );
          setNextToken(next);
        }
        setLoadingMore(false);
      })
      .catch((cause: unknown) => {
        if (coordinate !== coordinateRef.current) return;
        seenTokensRef.current.delete(requestedToken);
        setError(cause instanceof Error ? cause : new Error(String(cause)));
        setLoadingMore(false);
      })
      .finally(() => {
        if (coordinate === coordinateRef.current) {
          inFlightTokensRef.current.delete(requestedToken);
        }
      });
  }, [loading, loadingMore, nextToken, uniqueItems]);

  const reload = useCallback(() => {
    coordinateRef.current += 1;
    setGeneration((value) => value + 1);
  }, []);

  return {
    items,
    error,
    loading,
    loadingMore,
    hasMore: nextToken !== "",
    loadMore,
    reload,
  };
}
