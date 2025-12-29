export type SseEvent = {
  data: string;
  event?: string;
  id?: string;
  retry?: number;
};

export type ConsumeSseOptions = {
  doneSignal?: string;
  onDone?: () => void;
};

export type ConsumeSseJsonOptions = ConsumeSseOptions & {
  onParseError?: (error: unknown, raw: string) => void;
};

const parseSseEvent = (block: string): SseEvent | null => {
  if (!block.trim()) return null;
  const lines = block.split('\n');
  const dataLines: string[] = [];
  let event: string | undefined;
  let id: string | undefined;
  let retry: number | undefined;

  for (const line of lines) {
    if (!line || line.startsWith(':')) continue;
    const [field, ...rest] = line.split(':');
    const value = rest.join(':').replace(/^ /, '');
    if (field === 'data') {
      dataLines.push(value);
    } else if (field === 'event') {
      event = value;
    } else if (field === 'id') {
      id = value;
    } else if (field === 'retry') {
      const retryValue = Number(value);
      if (!Number.isNaN(retryValue)) retry = retryValue;
    }
  }

  const data = dataLines.join('\n');
  if (!data && !event && !id && retry === undefined) return null;
  return { data, event, id, retry };
};

export const consumeSse = async (
  response: Response,
  onEvent: (event: SseEvent) => void,
  options: ConsumeSseOptions = {}
): Promise<void> => {
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('无法创建流读取器');
  }

  const decoder = new TextDecoder();
  let buffer = '';
  const doneSignal = options.doneSignal ?? '[DONE]';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      buffer = buffer.replace(/\r\n/g, '\n');
      buffer = buffer.replace(/\r/g, '\n');

      const blocks = buffer.split('\n\n');
      buffer = blocks.pop() || '';

      for (const block of blocks) {
        const event = parseSseEvent(block);
        if (!event) continue;
        if (event.data === doneSignal) {
          options.onDone?.();
          return;
        }
        onEvent(event);
      }
    }

    if (buffer.trim()) {
      const event = parseSseEvent(buffer);
      if (event) {
        if (event.data === doneSignal) {
          options.onDone?.();
          return;
        }
        onEvent(event);
      }
    }
  } finally {
    reader.releaseLock();
  }

  options.onDone?.();
};

export const consumeSseJson = async <T>(
  response: Response,
  onData: (data: T) => void,
  options: ConsumeSseJsonOptions = {}
): Promise<void> => {
  await consumeSse(
    response,
    (event) => {
      if (!event.data) return;
      try {
        onData(JSON.parse(event.data) as T);
      } catch (error) {
        options.onParseError?.(error, event.data);
      }
    },
    options
  );
};
