"use client";

import { useState } from "react";
import { ImageOff } from "lucide-react";

import { getSampleImageUrl } from "@/lib/api";

// Renders a camera frame straight from the API's raw JPEG endpoint
// (/samples/{key}/image/cam_{n}). The endpoint streams image bytes, so a
// plain <img> is used (no JSON fetch, no Next image optimizer).
export function CameraImage({
  dataset,
  shard,
  sampleKey,
  cam,
  className,
}: {
  dataset: string;
  shard: string;
  sampleKey: string;
  cam: number;
  className?: string;
}) {
  const [failed, setFailed] = useState(false);
  const src = getSampleImageUrl(dataset, shard, sampleKey, cam);

  if (failed) {
    return (
      <div
        className={`flex items-center justify-center bg-slate-900 text-slate-600 ${className ?? "aspect-video w-full"}`}
      >
        <ImageOff className="size-5" />
      </div>
    );
  }

  return (
    <div
      className={`relative overflow-hidden bg-slate-900 ${className ?? "aspect-video w-full"}`}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={src}
        alt={`cam_${cam} of ${sampleKey}`}
        loading="lazy"
        className="absolute inset-0 h-full w-full object-contain bg-slate-950"
        onError={() => setFailed(true)}
      />
    </div>
  );
}
