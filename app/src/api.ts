import { Box, Health, PromptPoint, SessionResult } from './types';

export const API_URL =
  process.env.EXPO_PUBLIC_API_URL ?? 'http://127.0.0.1:8000';

async function read<T>(response: Response): Promise<T> {
  if (response.ok) {
    return response.json() as Promise<T>;
  }
  let message = `Request failed (${response.status})`;
  try {
    const data = await response.json();
    if (typeof data.detail === 'string') {
      message = data.detail;
    }
  } catch {
    // Keep the status message when the server has no JSON error body.
  }
  throw new Error(message);
}

export async function getHealth(): Promise<Health> {
  return read(await fetch(`${API_URL}/api/health`));
}

export async function createSession(file: File): Promise<SessionResult> {
  const form = new FormData();
  form.append('file', file);
  return read(
    await fetch(`${API_URL}/api/sessions`, {
      method: 'POST',
      body: form,
    }),
  );
}

export async function addPoint(
  sessionId: string,
  point: PromptPoint,
  positive: boolean,
): Promise<SessionResult> {
  return read(
    await fetch(`${API_URL}/api/sessions/${sessionId}/points`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ point, positive }),
    }),
  );
}

export async function addBox(
  sessionId: string,
  box: Box,
  positive: boolean,
): Promise<SessionResult> {
  return read(
    await fetch(`${API_URL}/api/sessions/${sessionId}/prompts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ box, positive }),
    }),
  );
}

export async function excludeObject(
  sessionId: string,
  objectId: number,
): Promise<SessionResult> {
  return read(
    await fetch(
      `${API_URL}/api/sessions/${sessionId}/objects/${objectId}/exclude`,
      { method: 'POST' },
    ),
  );
}

export async function refineResults(sessionId: string): Promise<SessionResult> {
  return read(
    await fetch(`${API_URL}/api/sessions/${sessionId}/refine`, {
      method: 'POST',
    }),
  );
}

export async function undoPrompt(sessionId: string): Promise<SessionResult> {
  return read(
    await fetch(`${API_URL}/api/sessions/${sessionId}/prompts/last`, {
      method: 'DELETE',
    }),
  );
}

export async function updatePoint(
  sessionId: string,
  index: number,
  point: PromptPoint,
): Promise<SessionResult> {
  return read(
    await fetch(`${API_URL}/api/sessions/${sessionId}/points/${index}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ point }),
    }),
  );
}

export async function updateBox(
  sessionId: string,
  index: number,
  box: Box,
): Promise<SessionResult> {
  return read(
    await fetch(`${API_URL}/api/sessions/${sessionId}/prompts/${index}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ box }),
    }),
  );
}

export async function deletePoint(
  sessionId: string,
  index: number,
): Promise<SessionResult> {
  return read(
    await fetch(`${API_URL}/api/sessions/${sessionId}/points/${index}`, {
      method: 'DELETE',
    }),
  );
}

export async function deletePoints(
  sessionId: string,
  indices: number[],
): Promise<SessionResult> {
  return read(
    await fetch(`${API_URL}/api/sessions/${sessionId}/points/delete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ indices }),
    }),
  );
}
