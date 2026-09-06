import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    globals: true,
    environment: "node",
    include: ["tests/**/*.test.ts"],
    // The adapter suites load the LangChain/LangGraph peer packages through
    // dynamic imports. Those imports are slow enough that the 5s default
    // makes whichever test happens to pull a package in first time out on a
    // loaded machine, so the failure rotates between suites instead of
    // pointing at a real defect. Generous enough to absorb that, still tight
    // enough to catch a genuine hang.
    testTimeout: 30_000,
    coverage: {
      provider: "v8",
      reporter: ["text", "json", "html"],
      include: ["src/**/*.ts"],
    },
  },
});
