// The version the sidebar foot shows. The source of truth is frontend/package.json "version"
// (version.test.ts keeps it equal to the repository's VERSION file). vite.config.ts reads it at
// build time and defines __APP_VERSION__, so no component writes the number by hand. Until
// ENH-004 the sidebar carried its own literal '0.5.0', which nothing tied to package.json.
//
// Origin: miDataworks frontend/src/config/version.ts @ 32cf406 (ENH-004).

declare const __APP_VERSION__: string;

export const APP_VERSION: string = __APP_VERSION__;
