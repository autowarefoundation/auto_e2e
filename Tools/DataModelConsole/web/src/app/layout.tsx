import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

import { Header } from "@/components/header";
import { Sidebar } from "@/components/sidebar";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "DataModelConsole",
  description:
    "Autonomous driving data and model intelligence console for the auto-e2e platform",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${geistMono.variable} bg-slate-900 text-slate-50 antialiased`}
      >
        <a
          href="#main-content"
          className="sr-only z-[60] rounded-md bg-slate-50 px-3 py-2 text-sm font-medium text-slate-950 focus:not-sr-only focus:fixed focus:top-3 focus:left-3"
        >
          Skip to main content
        </a>
        <Sidebar />
        <div className="md:pl-56">
          <Header />
          <main
            id="main-content"
            tabIndex={-1}
            className="mx-auto max-w-7xl p-6"
          >
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
