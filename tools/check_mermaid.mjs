// Validate every ```mermaid block in the given Markdown files.
// Usage: (cd tools && npm install) && node tools/check_mermaid.mjs $(git ls-files "*.md")
import { JSDOM } from 'jsdom';
import fs from 'fs';
const dom = new JSDOM('<!doctype html><html><body></body></html>');
globalThis.window = dom.window; globalThis.document = dom.window.document;
globalThis.DOMPurify = undefined;
const mermaid = (await import('mermaid')).default;
mermaid.initialize({startOnLoad:false});
let bad=0, n=0;
for (const f of process.argv.slice(2)) {
  const s = fs.readFileSync(f,'utf8');
  for (const m of s.matchAll(/```mermaid\n([\s\S]*?)```/g)) {
    n++;
    try { await mermaid.parse(m[1]); } catch(e) { bad++; console.log('FAIL', f, '\n', m[1].split('\n')[0], '\n', String(e.message).slice(0,300)); }
  }
}
console.log(`checked ${n}, failed ${bad}`);
