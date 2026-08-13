import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [payloadPath, outputDir, qaDir] = process.argv.slice(2);
if (!payloadPath || !outputDir || !qaDir) {
  throw new Error("usage: p10_catalogue_xlsx.mjs PAYLOAD OUTPUT_DIR QA_DIR");
}

const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(qaDir, { recursive: true });

const normalize = (value) => {
  if (value === null || value === undefined) return null;
  if (["string", "number", "boolean"].includes(typeof value)) return value;
  return JSON.stringify(value);
};

const columnName = (index) => {
  let n = index + 1;
  let result = "";
  while (n > 0) {
    const remainder = (n - 1) % 26;
    result = String.fromCharCode(65 + remainder) + result;
    n = Math.floor((n - 1) / 26);
  }
  return result;
};

async function buildWorkbook(spec) {
  const workbook = Workbook.create();
  const rendered = [];
  for (const sheetSpec of spec.sheets) {
    const sheet = workbook.worksheets.add(sheetSpec.name.slice(0, 31));
    const headers = sheetSpec.headers;
    const rows = sheetSpec.rows.map((row) => headers.map((header) => normalize(row[header])));
    const matrix = [headers, ...rows];
    const endColumn = columnName(Math.max(0, headers.length - 1));
    const endRow = Math.max(1, matrix.length);
    const range = sheet.getRange(`A1:${endColumn}${endRow}`);
    range.values = matrix;
    sheet.getRange(`A1:${endColumn}1`).format = {
      fill: "#1F4E78",
      font: { bold: true, color: "#FFFFFF" },
      wrapText: true,
      rowHeight: 32,
    };
    if (endRow > 1) {
      sheet.getRange(`A2:${endColumn}${endRow}`).format = {
        verticalAlignment: "top",
        wrapText: true,
      };
      sheet.tables.add(`A1:${endColumn}${endRow}`, true, `T_${sheetSpec.name.replace(/[^A-Za-z0-9]/g, "_").slice(0, 20)}`);
    }
    sheet.freezePanes.freezeRows(1);
    sheet.freezePanes.freezeColumns(Math.min(2, headers.length));
    range.format.autofitColumns();
    const visibleColumns = Math.min(headers.length, 12);
    for (let column = 0; column < headers.length; column += 1) {
      const col = columnName(column);
      const width = column < visibleColumns ? 22 : 16;
      sheet.getRange(`${col}1:${col}${endRow}`).format.columnWidth = width;
    }
    const previewEndColumn = columnName(Math.max(0, Math.min(headers.length, 12) - 1));
    const previewEndRow = Math.min(endRow, 30);
    const preview = await workbook.render({
      sheetName: sheetSpec.name.slice(0, 31),
      range: `A1:${previewEndColumn}${previewEndRow}`,
      scale: 1,
      format: "png",
    });
    const previewName = `${spec.output.replace(/\.xlsx$/i, "")}_${sheetSpec.name.replace(/[^A-Za-z0-9]/g, "_")}.png`;
    await fs.writeFile(path.join(qaDir, previewName), new Uint8Array(await preview.arrayBuffer()));
    rendered.push(previewName);
  }
  const keySheet = spec.sheets[0];
  const keyEndColumn = columnName(Math.max(0, Math.min(keySheet.headers.length, 12) - 1));
  const keyInspection = await workbook.inspect({
    kind: "table",
    range: `${keySheet.name.slice(0, 31)}!A1:${keyEndColumn}${Math.min(keySheet.rows.length + 1, 20)}`,
    include: "values,formulas",
    tableMaxRows: 20,
    tableMaxCols: 12,
  });
  const errorInspection = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 300 },
    summary: "final formula error scan",
  });
  const xlsx = await SpreadsheetFile.exportXlsx(workbook);
  const outputPath = path.join(outputDir, spec.output);
  await xlsx.save(outputPath);
  return {
    output: spec.output,
    sheet_count: spec.sheets.length,
    sheets: spec.sheets.map((sheet) => ({ name: sheet.name, row_count: sheet.rows.length, column_count: sheet.headers.length })),
    rendered_previews: rendered,
    key_inspection: keyInspection.ndjson,
    formula_error_scan: errorInspection.ndjson,
  };
}

const results = [];
for (const workbookSpec of payload.workbooks) {
  results.push(await buildWorkbook(workbookSpec));
}
await fs.writeFile(
  path.join(outputDir, "PRIVATE_XLSX_QA.json"),
  `${JSON.stringify({ status: "PASS", artifact_tool: "@oai/artifact-tool", workbooks: results }, null, 2)}\n`,
  "utf8",
);
