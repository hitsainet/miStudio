import js from '@eslint/js';
import globals from 'globals';
import reactHooks from 'eslint-plugin-react-hooks';
import reactRefresh from 'eslint-plugin-react-refresh';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  { ignores: ['dist'] },
  {
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
    },
    plugins: {
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      // REACT COMPILER DIAGNOSTICS AS WARNINGS (2026-10-21, eslint-plugin-react-hooks 5 -> 7).
      //
      // v7's `recommended` set adds the React Compiler's own checks as ERRORS. This app does not
      // build with the React Compiler, and on the day of the bump they reported 111 sites that
      // already existed — none is a hooks-order crash (that is `rules-of-hooks`, still an error):
      //   set-state-in-effect 57, static-components 25 (components declared inside a component,
      //   so they remount on every parent render), refs 14, immutability 7 (5 of them a `const`
      //   arrow used by an effect above its declaration: legal, the effect runs after render),
      //   use-memo 4, purity 3, preserve-manual-memoization 1.
      // Lint is a CI gate on errors (frontend-ci.yml, MIS-E2E-024), so as errors they would block
      // every frontend change on a 111-site refactor unrelated to it. They stay VISIBLE as
      // warnings, named one by one here (not the whole plugin), and are listed with file and line
      // as debt in 0xcc/reviews/deps_sync_2026-10-21.md. Promote a rule back to 'error' once its
      // count is zero.
      'react-hooks/set-state-in-effect': 'warn',
      'react-hooks/static-components': 'warn',
      'react-hooks/refs': 'warn',
      'react-hooks/immutability': 'warn',
      'react-hooks/use-memo': 'warn',
      'react-hooks/purity': 'warn',
      'react-hooks/preserve-manual-memoization': 'warn',
      'react-refresh/only-export-components': [
        'warn',
        { allowConstantExport: true },
      ],
      // The `_` prefix is this codebase's "deliberately unused" convention, and
      // it was only honoured for ARGUMENTS. Variables, caught errors and
      // destructuring rest-siblings were still errors — including
      // `const { [k]: _, ...rest }`, where the binding is REQUIRED syntax for
      // omitting a key and cannot be removed at all.
      //
      // Lint has never run in CI (MIS-E2E-024), so none of this was visible.
      // Turning it on without this is how a mechanical clean-up ends up
      // deleting bindings the code needs.
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_',
          destructuredArrayIgnorePattern: '^_',
          ignoreRestSiblings: true,
        },
      ],
      '@typescript-eslint/no-explicit-any': 'warn',
    },
  },
);
