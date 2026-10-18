import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['test/integration.test.ts', 'test/integration-passage.test.ts'],
    testTimeout: 120_000,
  },
});
