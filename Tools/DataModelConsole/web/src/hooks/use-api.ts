"use client";

// Small client-side data fetching hook. All Phase 1 data is fetched from the
// Go API in the browser (no server actions), so a hook + Skeleton pattern is
// used for loading states.

import { useCallback, useEffect, useRef, useState } from "react";

export interface UseApiResult<T> {
  data: T | null;
  error: Error | null;
  loading: boolean;
  reload: () => void;
}

export function useApi<T>(
  fetcher: () => Promise<T>,
  deps: readonly unknown[] = [],
  enabled = true,
): UseApiResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);
  const [generation, setGeneration] = useState(0);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  useEffect(() => {
    if (!enabled) {
      setData(null);
      setError(null);
      setLoading(false);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetcherRef
      .current()
      .then((d) => {
        if (!cancelled) {
          setData(d);
          setLoading(false);
        }
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setError(e instanceof Error ? e : new Error(String(e)));
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, enabled, generation]);

  const reload = useCallback(() => setGeneration((g) => g + 1), []);

  return { data, error, loading, reload };
}
