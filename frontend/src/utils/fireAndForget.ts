/**
 * Start a store action without awaiting it, when its failure is already
 * surfaced elsewhere.
 *
 * Most async actions in this app's stores end their catch block by recording a
 * message in the store (`featuresError`, `trainingsError`, …) and then
 * RE-THROWING. That is right for the 87 call sites that await the action and
 * want to know whether it succeeded — a form submit that must stay open on
 * failure, for instance.
 *
 * It is wrong for a call made purely to refresh what is on screen. Those
 * ignore the returned promise, so the re-thrown error has no consumer and
 * becomes an unhandled promise rejection in the user's browser: console noise
 * on every flaky request, error-reporter noise, and in some setups a dev
 * overlay — all for a failure the UI is already displaying from store state.
 *
 * Wrapping the call says that out loud and is greppable, where a bare
 * `void promise.catch(() => {})` at twenty call sites reads like a mistake
 * someone should clean up.
 *
 * WHAT THIS DOES NOT DO: it never decides that an error is unimportant. The
 * action has already put the message where the UI reads it. This only declines
 * to raise the same error a second time, to nobody.
 */
export function fireAndForget(promise: Promise<unknown> | undefined): void {
  if (!promise || typeof promise.catch !== 'function') return;
  void promise.catch(() => {
    // Intentionally empty — see the note above. The store recorded it.
  });
}
