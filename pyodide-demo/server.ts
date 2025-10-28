import { serve } from "bun";
const base = new URL("./dist/", import.meta.url).pathname;

serve({
  port: 3000,
  fetch(req) {
    const url = new URL(req.url);
    let pathname = url.pathname;
    if (pathname === "/") pathname = "/index.html";
    try {
      const file = Bun.file(base + pathname);
      return new Response(file, { headers: { "Content-Type": file.type } });
    } catch {
      return new Response("Not found", { status: 404 });
    }
  },
});
