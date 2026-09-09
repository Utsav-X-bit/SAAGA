// Build SAAGA conference paper DOCX (two-column, ICML/HarmBench-style layout)
const fs = require('fs');
const path = require('path');
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
        AlignmentType, WidthType, BorderStyle, HeadingLevel, PageBreak, TabStopType, TabStopPosition, SectionType } = require('docx');

const DIR = __dirname;
const md = fs.readFileSync(path.join(DIR, 'SAAGA_full.md'), 'utf8');

const FONT = 'Times New Roman';
const B = (text, opts = {}) => new TextRun({ text, font: FONT, size: 20, ...opts }); // 10pt
const S = (text, opts = {}) => new TextRun({ text, font: FONT, size: 18, ...opts }); // 9pt

// ---- inline parser: **bold**, *italic*
function runs(text, base = {}) {
  const out = [];
  text.split('**').forEach((part, idx) => {
    const isBold = idx % 2 === 1;
    part.split('*').forEach((ipart, jdx) => {
      if (!ipart) return;
      out.push(B(ipart, Object.assign({}, base, isBold ? { bold: true } : {}, jdx % 2 === 1 ? { italics: true } : {})));
    });
  });
  return out;
}

// ---- figure registry: file -> {wIn, hIn}
const figs = {
  'fig1_pipeline.png': { wIn: 6.7, full: true },
  'fig2_main.png':     { wIn: 3.35, full: false },
  'fig6_autored_cmp.png': { wIn: 3.35, full: false },
  'fig4_defense.png':  { wIn: 3.35, full: false },
  'fig5_memory.png':   { wIn: 6.7, full: true },
  'fig6_autored_cmp.png': { wIn: 3.35, full: false },
};
function figBlock(file, caption) {
  const info = figs[file];
  const img = fs.readFileSync(path.join(DIR, 'figs', file));
  // read PNG dims
  const w = img.readUInt32BE(16), h = img.readUInt32BE(20);
  const wpx = Math.round(info.wIn * 96), hpx = Math.round(wpx * h / w);
  const cap = new Paragraph({
    spacing: { before: 80, after: 200 },
    children: [new TextRun({ text: caption, font: FONT, size: 18, italics: false })],
    alignment: AlignmentType.CENTER,
  });
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 200, after: 0 },
      children: [new ImageRun({ type: 'png', data: img, transformation: { width: wpx, height: hpx } })],
    }),
    cap,
  ];
}

// ---- table builder (markdown pipe table)
const thin = { style: BorderStyle.SINGLE, size: 4, color: '000000' };
const thick = { style: BorderStyle.SINGLE, size: 10, color: '000000' };
const none = { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' };
function mdTable(lines) {
  const rows = lines.map(l => l.trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim()))
    .filter(r => !r.every(c => /^:?-+:?$/.test(c)));
  const ncol = rows[0].length;
  const totalW = 4464; // one column of the two-column body
  const colW = Math.floor(totalW / ncol);
  return makeTable(rows, totalW, colW);
}
function makeTable(rows, totalW, colW) {
  return new Table({
    width: { size: totalW, type: WidthType.DXA },
    columnWidths: Array(rows[0].length).fill(colW),
    borders: { top: thick, bottom: thick, left: none, right: none, insideHorizontal: none, insideVertical: none },
    rows: rows.map((r, ri) => new TableRow({
      tableHeader: ri === 0,
      children: r.map(c => new TableCell({
        width: { size: colW, type: WidthType.DXA },
        borders: { top: ri === 1 ? thin : none, bottom: none, left: none, right: none },
        children: [new Paragraph({
          alignment: ri === 0 ? AlignmentType.CENTER : (/[0-9.,%/-]+/.test(c) && !/[a-z]/.test(c) ? AlignmentType.CENTER : AlignmentType.LEFT),
          spacing: { before: 20, after: 20 },
          children: [new TextRun({ text: c, font: FONT, size: 17, bold: ri === 0 })],
        })],
      })),
    })),
  });
}

// ---- parse markdown into sections (body content)
const lines = md.split('\n');
let title = '', abstractText = '';
const bodyLines = [];
let mode = 'pre'; // pre | abstract | body
for (const line of lines) {
  if (line.startsWith('# ') && mode === 'pre') { title = line.slice(2).trim(); mode = 'abstract'; continue; }
  if (line.startsWith('## Abstract')) { mode = 'abstract'; continue; }
  if (mode === 'abstract') {
    if (line.startsWith('## ')) { mode = 'body'; bodyLines.push(line); }
    else if (line.trim()) abstractText += line.trim() + ' ';
    continue;
  }
  bodyLines.push(line);
}

const PAGE = {
  size: { width: 12240, height: 15840 },
  margin: { top: 1440, bottom: 1440, left: 1440, right: 1440, header: 720, footer: 720 },
};
const twoColProps = {
  type: SectionType.CONTINUOUS,
  column: { count: 2, space: 432 },
  page: PAGE,
};
const oneColProps = {
  type: SectionType.CONTINUOUS,
  column: { count: 1, space: 0 },
  page: PAGE,
};

// Build content list; split into sections when full-width figures appear
const sections = [];
let cur = { properties: twoColProps, children: [] };
function flush() { if (cur.children.length) sections.push(cur); cur = { properties: twoColProps, children: [] }; }

let i = 0;
const refStart = bodyLines.findIndex(l => l.startsWith('## References'));
const appStart = bodyLines.findIndex(l => l.startsWith('## Appendix'));
while (i < bodyLines.length) {
  const line = bodyLines[i];
  const figm = line.match(/^\[\[FIG:(\S+)\|(.+)\]\]$/);
  if (figm) {
    const full = figs[figm[1]].full;
    if (full) {
      const kids = figBlock(figm[1], figm[2]);
      flush();
      sections.push({ properties: oneColProps, children: kids });
      cur = { properties: twoColProps, children: [] };
    } else {
      cur.children.push(...figBlock(figm[1], figm[2]));
    }
    i++; continue;
  }
  if (line.startsWith('|')) {
    const block = [];
    while (i < bodyLines.length && bodyLines[i].startsWith('|')) { block.push(bodyLines[i]); i++; }
    // wide tables (>=6 cols) get their own full-width section
    const ncol = block[0].split('|').length - 2;
    const wide = ncol >= 6;
    if (wide) {
      const t = mdTableFull(block);
      flush();
      sections.push({ properties: oneColProps, children: [t, new Paragraph({ spacing: { after: 120 }, children: [] })] });
      cur = { properties: twoColProps, children: [] };
    } else {
      cur.children.push(new Paragraph({ spacing: { before: 120, after: 0 }, children: [] }), mdTable(block), new Paragraph({ spacing: { after: 120 }, children: [] }));
    }
    continue;
  }
  if (!line.trim()) { i++; continue; }
  if (line.startsWith('### ')) {
    cur.children.push(new Paragraph({ spacing: { before: 160, after: 60 }, children: [new TextRun({ text: line.slice(4), font: FONT, size: 20, bold: true, italics: true })] }));
  } else if (line.startsWith('## ')) {
    cur.children.push(new Paragraph({ spacing: { before: 240, after: 80 }, children: [new TextRun({ text: line.slice(3), font: FONT, size: 21, bold: true })] }));
  } else if (line.startsWith('1. ') || line.startsWith('2. ') || line.startsWith('3. ') || line.startsWith('4. ')) {
    const m = line.match(/^(\d)\. (.*)$/);
    cur.children.push(new Paragraph({ spacing: { after: 60 }, indent: { left: 360, hanging: 300 }, alignment: AlignmentType.JUSTIFIED, children: [B(`${m[1]}. `, { bold: true }), ...runs(m[2])] }));
  } else if (refStart !== -1 && appStart !== -1 && i > refStart && i < appStart) {
    cur.children.push(new Paragraph({ spacing: { after: 60 }, indent: { left: 360, hanging: 360 }, alignment: AlignmentType.JUSTIFIED, children: [S(line.trim())] }));
  } else {
    cur.children.push(new Paragraph({ spacing: { after: 0, line: 260 }, indent: { firstLine: 200 }, alignment: AlignmentType.JUSTIFIED, children: runs(line.trim()) }));
  }
  i++;
}
flush();

function mdTableFull(blockLines) {
  const rows = blockLines.map(l => l.trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim()))
    .filter(r => !r.every(c => /^:?-+:?$/.test(c)));
  const ncol = rows[0].length;
  const totalW = 9360;
  const colW = Math.floor(totalW / ncol);
  return new Table({
    width: { size: totalW, type: WidthType.DXA },
    columnWidths: Array(ncol).fill(colW),
    borders: { top: thick, bottom: thick, left: none, right: none, insideHorizontal: none, insideVertical: none },
    rows: rows.map((r, ri) => new TableRow({
      tableHeader: ri === 0,
      children: r.map((c, ci) => new TableCell({
        width: { size: colW, type: WidthType.DXA },
        borders: { top: ri === 1 ? thin : none, bottom: none, left: none, right: none },
        children: [new Paragraph({
          alignment: ri === 0 ? AlignmentType.CENTER : (/[0-9.,%/-]/.test(c) && !/[a-z]/.test(c) ? AlignmentType.CENTER : AlignmentType.LEFT),
          spacing: { before: 30, after: 30 },
          children: [new TextRun({ text: c, font: FONT, size: 18, bold: ri === 0 })],
        })],
      })),
    })),
  });
}

// ---- title + abstract section
const titleSection = {
  properties: oneColProps,
  children: [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: title, font: FONT, size: 34, bold: true })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: 'Anonymous Authors', font: FONT, size: 22 })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 300 }, children: [new TextRun({ text: 'Double-blind submission', font: FONT, size: 18, italics: true })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 100 }, children: [new TextRun({ text: 'Abstract', font: FONT, size: 20, bold: true })] }),
    new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 0, line: 260 }, indent: { firstLine: 240 }, children: [new TextRun({ text: abstractText.trim(), font: FONT, size: 19, italics: true })] }),
  ],
};

const doc = new Document({
  creator: 'SAAGA authors',
  title: title,
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  sections: [titleSection, ...sections],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(path.join(DIR, 'SAAGA_paper.docx'), buf);
  console.log('DOCX written:', buf.length, 'bytes');
});
