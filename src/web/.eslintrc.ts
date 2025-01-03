module.exports = {
  root: true,
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 2022,
    sourceType: 'module',
    ecmaFeatures: {
      jsx: true
    },
    project: './tsconfig.json'
  },
  settings: {
    react: {
      version: '18.2' // @version ^18.2.0
    }
  },
  env: {
    browser: true,
    es2022: true,
    node: true,
    jest: true
  },
  plugins: [
    '@typescript-eslint', // @version ^5.59.0
    'react', // @version ^7.32.2
    'react-hooks' // @version ^4.6.0
  ],
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:@typescript-eslint/recommended-requiring-type-checking',
    'plugin:react/recommended',
    'plugin:react-hooks/recommended',
    'prettier' // @version ^8.8.0
  ],
  rules: {
    // TypeScript specific rules
    '@typescript-eslint/explicit-function-return-type': 'error',
    '@typescript-eslint/no-explicit-any': 'error',
    '@typescript-eslint/no-unused-vars': 'error',
    '@typescript-eslint/strict-boolean-expressions': 'error',
    '@typescript-eslint/no-unsafe-member-access': 'error',
    '@typescript-eslint/no-floating-promises': 'error',
    '@typescript-eslint/await-thenable': 'error',
    '@typescript-eslint/no-misused-promises': 'error',
    '@typescript-eslint/no-unnecessary-type-assertion': 'error',
    '@typescript-eslint/prefer-as-const': 'error',
    '@typescript-eslint/no-non-null-assertion': 'error',
    '@typescript-eslint/member-ordering': ['error', {
      default: [
        'static-field',
        'instance-field',
        'constructor',
        'static-method',
        'instance-method'
      ]
    }],

    // React specific rules
    'react/react-in-jsx-scope': 'off',
    'react/prop-types': 'off',
    'react/jsx-no-target-blank': 'error',
    'react/jsx-curly-brace-presence': ['error', { props: 'never', children: 'never' }],
    'react/no-array-index-key': 'error',
    'react/no-unused-prop-types': 'error',
    'react/jsx-boolean-value': ['error', 'never'],
    'react/jsx-no-useless-fragment': 'error',
    'react/no-unescaped-entities': 'error',

    // React Hooks rules
    'react-hooks/rules-of-hooks': 'error',
    'react-hooks/exhaustive-deps': 'warn',

    // General rules
    'no-console': ['warn', { allow: ['warn', 'error'] }],
    'no-debugger': 'error',
    'no-alert': 'error',
    'no-var': 'error',
    'prefer-const': 'error',
    'eqeqeq': ['error', 'always'],
    'no-multiple-empty-lines': ['error', { max: 1, maxEOF: 0 }],
    'no-nested-ternary': 'error',
    'no-duplicate-imports': 'error',
    'sort-imports': ['error', {
      ignoreCase: true,
      ignoreDeclarationSort: true
    }]
  },
  overrides: [
    {
      files: ['**/*.test.ts', '**/*.test.tsx'],
      env: {
        jest: true
      },
      rules: {
        '@typescript-eslint/no-explicit-any': 'off',
        '@typescript-eslint/no-non-null-assertion': 'off'
      }
    },
    {
      files: ['**/lidar/**/*.ts', '**/lidar/**/*.tsx'],
      rules: {
        '@typescript-eslint/no-unsafe-member-access': 'warn',
        '@typescript-eslint/no-floating-promises': 'warn'
      }
    },
    {
      files: ['**/fleet/**/*.ts', '**/fleet/**/*.tsx'],
      rules: {
        '@typescript-eslint/no-misused-promises': 'warn'
      }
    }
  ],
  ignorePatterns: [
    'node_modules',
    'build',
    'dist',
    'coverage',
    'vite.config.ts'
  ]
};