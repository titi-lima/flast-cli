# demo-vitest

Small Vitest project for exercising the FLAST Vitest provider.

```bash
npm install
flast init vitest
flast observe vitest --runs 3 --framework-arg tests/sample.test.js
flast train
flast diagnostics vitest --format json --framework-arg tests/sample.test.js
```
