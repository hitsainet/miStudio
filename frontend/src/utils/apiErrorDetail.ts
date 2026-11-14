/**
 * The reason an API request failed, as the backend gave it (review R2, B6).
 *
 * FastAPI answers a refusal with `{ detail: string }` and a request-validation
 * failure with `{ detail: [{ loc, msg, type }, ...] }`. Callers read
 * `error.response.data.message`, which FastAPI never sends, and then alerted
 * axios's own `error.message` — so a training refused because its extraction
 * holds no MLP activations reached the user as
 * "Request failed with status code 422", with the reason thrown away.
 *
 * Order: `detail` (a string, a list of validation errors, or an object with a
 * message), then a `message` field, then `fallback`.
 */
export function apiErrorDetail(error: unknown, fallback: string): string {
  const data = (error as { response?: { data?: unknown } } | null | undefined)?.response?.data as
    | { detail?: unknown; message?: unknown }
    | undefined;
  const detail = data?.detail;

  if (typeof detail === 'string' && detail.trim()) return detail;

  if (Array.isArray(detail) && detail.length > 0) {
    return detail.map(validationErrorText).join('; ');
  }

  if (detail && typeof detail === 'object') {
    const message = (detail as { message?: unknown }).message;
    if (typeof message === 'string' && message.trim()) return message;
    return JSON.stringify(detail);
  }

  if (typeof data?.message === 'string' && data.message.trim()) return data.message;

  return fallback;
}

function validationErrorText(entry: unknown): string {
  if (typeof entry === 'string') return entry;
  if (!entry || typeof entry !== 'object') return String(entry);
  const { loc, msg } = entry as { loc?: unknown; msg?: unknown };
  const text = typeof msg === 'string' ? msg : JSON.stringify(entry);
  // FastAPI prefixes every body field's location with "body"; it names no field.
  const where = Array.isArray(loc) ? loc.filter((part) => part !== 'body').join('.') : '';
  return where ? `${where}: ${text}` : text;
}
