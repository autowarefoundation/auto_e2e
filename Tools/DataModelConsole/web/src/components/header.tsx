export function Header() {
  return (
    <header className="sticky top-0 z-30 flex h-14 items-center border-b border-slate-800 bg-slate-950/80 px-6 backdrop-blur">
      <h1 className="text-sm font-semibold tracking-tight">
        DataModelConsole
      </h1>
      <span className="ml-3 rounded-full border border-slate-700 px-2 py-0.5 text-[10px] uppercase tracking-wider text-slate-400">
        auto-e2e platform
      </span>
    </header>
  );
}
