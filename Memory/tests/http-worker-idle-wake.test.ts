import type { Server } from "node:http";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createMemoryHttpServer, type LlmCompletionOptions, type LlmMessage } from "../src/index.js";
import { closeMemoryHttpServer } from "../src/server/http.js";
import { addAgentSourceImport, createBatchReflectionLlm, createMemoryServiceFixture } from "./fixtures/memory-service-fixture.js";

const fixture = createMemoryServiceFixture();
const servers: Server[] = [];
const namespace = { source: "codex", profileId: "idle-wake", userId: "idle-wake-user" };
afterEach(async () => {
  await Promise.all(servers.splice(0).map(closeMemoryHttpServer));
  await fixture.cleanup();
  vi.restoreAllMocks();
});
async function listen(server: Server): Promise<string> {
  servers.push(server);
  await new Promise<void>((resolve, reject) => { server.once("error", reject); server.listen(0, "127.0.0.1", resolve); });
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("expected local TCP address");
  return `http://127.0.0.1:${address.port}`;
}
async function wakeWorker(base: string, memoryIds: string[]) {
  const response = await fetch(`${base}/api/v1/worker/import-summaries/enqueue`, {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ memoryIds })
  });
  expect(response.status).toBe(200);
}
async function waitFor(check: () => boolean, timeoutMs = 3_000) {
  const deadline = Date.now() + timeoutMs;
  while (!check()) {
    if (Date.now() >= deadline) throw new Error("timed out waiting for the automatic worker");
    await new Promise(resolve => setTimeout(resolve, 5));
  }
}

// Real HTTP auto scheduling, MemoryService and SQLite. Only the model is an in-process substitute.
describe("HTTP automatic worker while idle", () => {
  it("waits for the next future deadline when a due job cannot run yet", async () => {
    let modelAvailable = false;
    const llm = createBatchReflectionLlm([], "Idle wake summary"), complete = llm.completeJson.bind(llm);
    llm.completeJson = async <T extends Record<string, unknown>>(messages: LlmMessage[], options: LlmCompletionOptions): Promise<T> => {
      if (!modelAvailable) throw new Error("request timed out");
      return complete<T>(messages, options);
    };
    const { service, db } = fixture.createTestService({ llm, skillLlm: llm });
    const memory = addAgentSourceImport(service, namespace, "summary fails once", "idle-wake-1");
    const summaryJob = () => db.db.prepare(
      "SELECT id, status, attempts, available_at FROM evolution_jobs WHERE job_type = 'import_summary' AND target_memory_id = ?"
    ).get(memory.id) as { id: string; status: string; attempts: number; available_at: string | null };
    const run = vi.spyOn(service, "runWorkerOnce");
    const base = await listen(createMemoryHttpServer({
      service, workerStartupFallbackMs: 60_000, workerIdlePollMs: 60_000, startAgentSourceAutomation: false
    }));

    // The summary fails once and waits in backoff; its memory stays summary_pending.
    await wakeWorker(base, [memory.id]);
    await waitFor(() => summaryJob().status === "failed");
    expect(service.memoryProcessingStatus([memory.id], { namespace }).items[0]?.state).toBe("summary_pending");

    // A reward whose own time has come, but which may not run before that summary exists.
    service["workerHandlers"].enqueueJob({
      jobType: "reward", userId: namespace.userId,
      payload: { l1MemoryId: memory.id, trigger: "idle-wake", targetKind: "episode", phase: "final",
        runAfter: new Date(Date.now() - 60_000).toISOString() }
    });
    // Shorten the backoff so the test can also observe the real wake-up.
    const retryAt = Date.now() + 800;
    db.db.prepare("UPDATE evolution_jobs SET available_at = ? WHERE id = ?").run(new Date(retryAt).toISOString(), summaryJob().id);
    // The earliest queue deadline now lies in the past and belongs to a job that cannot be claimed.
    expect(service.nextWorkerRunAt()).toBeLessThan(Date.now());

    modelAvailable = true;
    const before = run.mock.calls.length;
    await wakeWorker(base, []);
    await new Promise(resolve => setTimeout(resolve, 300));
    // Nothing can run before retryAt. One drain discovers that; the worker then sleeps until retryAt
    // instead of starting a new drain every millisecond for the reward that is due but blocked.
    expect(run.mock.calls.length - before).toBeLessThanOrEqual(3);
    expect(summaryJob()).toMatchObject({ status: "failed", attempts: 1 });

    // No further request arrives: the retry deadline alone must wake the worker.
    await waitFor(() => summaryJob().status === "succeeded");
    expect(summaryJob().attempts).toBe(2);
  }, 15_000);

  it("wakes an idle worker as soon as a producer enqueues work", async () => {
    // The idle poll is left at 60 s, so only the producer's own wake-up can finish this in time.
    const llm = createBatchReflectionLlm([], "Producer wake summary");
    const { service, db } = fixture.createTestService({ llm, skillLlm: llm });
    const base = await listen(createMemoryHttpServer({
      service, workerStartupFallbackMs: 60_000, workerIdlePollMs: 60_000, startAgentSourceAutomation: false
    }));
    const summaryStatus = (memoryId: string) => (db.db.prepare(
      "SELECT status FROM evolution_jobs WHERE job_type = 'import_summary' AND target_memory_id = ?"
    ).get(memoryId) as { status: string } | undefined)?.status;

    // The first summary leaves the worker with nothing to lease, so it goes idle.
    const warmUp = addAgentSourceImport(service, namespace, "warm-up turn", "idle-wake-2");
    await wakeWorker(base, [warmUp.id]);
    await waitFor(() => summaryStatus(warmUp.id) === "succeeded");

    const memory = addAgentSourceImport(service, namespace, "producer turn", "idle-wake-3");
    await wakeWorker(base, [memory.id]);
    await waitFor(() => summaryStatus(memory.id) === "succeeded");
  }, 15_000);
});
