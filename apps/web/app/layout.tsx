import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "BigBook Recommender",
  description: "A book recommendation interface powered by the BigBook API."
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="es">
      <body>{children}</body>
    </html>
  );
}
