import { redirect } from "next/navigation";

// next.config.ts redirects "/" before this renders; this is the fallback.
export default function Home() {
  redirect("/live/index.html");
}
