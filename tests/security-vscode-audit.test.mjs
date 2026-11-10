import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');

function walkFiles(dir, fileList = []) {
  if (!fs.existsSync(dir)) return fileList;
  try {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      if (entry.name === '.git' || entry.name === 'node_modules') continue;
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        walkFiles(fullPath, fileList);
      } else {
        fileList.push(fullPath);
      }
    }
  } catch {
    // Ignore inaccessible directories
  }
  return fileList;
}

const allFiles = walkFiles(ROOT);

test('No .vscode/tasks.json uses runOn: folderOpen or executes font files', () => {
  const taskFiles = allFiles.filter((f) => f.endsWith(path.join('.vscode', 'tasks.json')));
  for (const file of taskFiles) {
    const raw = fs.readFileSync(file, 'utf8');
    assert.doesNotMatch(
      raw,
      /"runOn"\s*:\s*"folderOpen"/i,
      `Forbidden auto-run task detected in ${file}`
    );
    assert.doesNotMatch(
      raw,
      /fa-solid\.lkf|public\/fonts/i,
      `Suspicious font execution target detected in ${file}`
    );
  }
});

test('No .vscode/settings.json enables automatic task execution', () => {
  const settingFiles = allFiles.filter((f) => f.endsWith(path.join('.vscode', 'settings.json')));
  for (const file of settingFiles) {
    const raw = fs.readFileSync(file, 'utf8');
    assert.doesNotMatch(
      raw,
      /"task\.allowAutomaticTasks"\s*:\s*("on"|true)/i,
      `Forbidden task.allowAutomaticTasks setting detected in ${file}`
    );
  }
});

test('No suspicious .lkf files or script payloads exist inside public/fonts/', () => {
  const fontFiles = allFiles.filter((f) => f.includes(path.join('public', 'fonts')));
  for (const file of fontFiles) {
    assert.ok(
      !file.endsWith('.lkf'),
      `Forbidden .lkf payload file detected: ${file}`
    );
    const buf = fs.readFileSync(file);
    const sample = buf.subarray(0, Math.min(buf.length, 4096)).toString('utf8');
    assert.doesNotMatch(
      sample,
      /\b(eval|Function|require|child_process|Buffer\.from|execSync|spawn)\b/,
      `Executable JavaScript patterns detected inside font asset: ${file}`
    );
  }
});
