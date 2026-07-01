const fs = require('fs');
const path = require('path');

const extensionRoot = path.resolve(__dirname, '..');
const cliRoot = path.resolve(extensionRoot, '..');
const stagedRoot = path.join(extensionRoot, 'flast-cli');

function copyPath(source, destination) {
  fs.cpSync(source, destination, {
    recursive: true,
    force: true,
    filter: (entry) => {
      const name = path.basename(entry);
      return ![
        '__pycache__',
        '.pytest_cache',
        '.mypy_cache',
        '.ruff_cache',
        '.venv',
      ].includes(name);
    },
  });
}

fs.rmSync(stagedRoot, { recursive: true, force: true });
fs.mkdirSync(stagedRoot, { recursive: true });

for (const name of ['flast_cli', 'pyproject.toml', 'setup.py', 'README.md']) {
  const source = path.join(cliRoot, name);
  if (fs.existsSync(source)) {
    copyPath(source, path.join(stagedRoot, name));
  }
}

console.log(`Staged FLAST CLI into ${stagedRoot}`);
