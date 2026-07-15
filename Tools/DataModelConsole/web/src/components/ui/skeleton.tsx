import { cn } from "@/lib/utils"

function Skeleton({
  className,
  label,
  ...props
}: React.ComponentProps<"div"> & { label?: string }) {
  return (
    <div
      data-slot="skeleton"
      role={label ? "status" : undefined}
      aria-live={label ? "polite" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
      className={cn("animate-pulse rounded-md bg-muted", className)}
      {...props}
    />
  )
}

export { Skeleton }
