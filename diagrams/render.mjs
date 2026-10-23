// render each diagram html to a 2x png cropped to <main>. uses the web-designer skill's playwright-core.
// usage: node diagrams/render.mjs <diagrams dir> <chrome or edge exe>
import { createRequire } from "module";
import { pathToFileURL } from "url";
import path from "path";
const require = createRequire(path.join(process.env.USERPROFILE, ".claude/skills/web-designer/scripts/x.js"));
const { chromium } = require("playwright-core");
const dir = process.argv[2];
const exe = process.argv[3];
const browser = await chromium.launch({ executablePath: exe });
const page = await browser.newPage({ viewport: { width: 1200, height: 600 }, deviceScaleFactor: 2 });
for (const name of ["how", "arch", "moves", "train", "results"]) {
  await page.goto(pathToFileURL(path.join(dir, name + ".html")).href, { waitUntil: "networkidle" });
  await page.evaluate(() => document.fonts.ready);
  await page.locator("main").screenshot({ path: path.join(dir, name + ".png") });
  console.log(name, "ok");
}
await browser.close();
