// Built by src/config/buildHardening.test.ts through the real vite.config.ts. Each call carries a
// distinct marker so the test can say which survived the production build.
export function run(payload: unknown): void {
  console.log('marker-log', payload);
  console.debug('marker-debug', payload);
  console.info('marker-info', payload);
  console.trace('marker-trace');
  console.warn('marker-warn', payload);
  console.error('marker-error', payload);
}
