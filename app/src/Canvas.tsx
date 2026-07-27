import { PointerEvent as WebPointerEvent, useRef, useState } from 'react';
import {
  GestureResponderEvent,
  Image,
  LayoutChangeEvent,
  Platform,
  StyleSheet,
  Text,
  View,
} from 'react-native';

import {
  Box,
  ImageSize,
  PromptMark,
  PromptPoint,
  PromptTool,
  ResultObject,
} from './types';

type Point = { x: number; y: number };
type BoxMode = 'move' | 'nw' | 'ne' | 'sw' | 'se';
type EditState =
  | {
      kind: 'point';
      index: number;
      origin: PromptPoint;
      original: PromptPoint;
      point: PromptPoint;
    }
  | {
      kind: 'box';
      index: number;
      mode: BoxMode;
      origin: PromptPoint;
      original: Box;
      box: Box;
    };

type Props = {
  uri: string | null;
  imageSize: ImageSize | null;
  prompts: PromptMark[];
  objects: ResultObject[];
  positive: boolean;
  disabled: boolean;
  interactive?: boolean;
  label: string;
  loadingText: string | null;
  tool: PromptTool;
  onPoint?: (point: PromptPoint) => void;
  onBox?: (box: Box) => void;
  onChangePrompt?: (index: number, point: PromptPoint) => void;
  onChangeBox?: (index: number, box: Box) => void;
  onChoose?: () => void;
  onObject?: (item: ResultObject) => void;
  onSelectPrompts?: (indices: number[]) => void;
  selectedPrompts: number[];
};

export function Canvas({
  uri,
  imageSize,
  prompts,
  objects,
  positive,
  disabled,
  interactive = true,
  label,
  loadingText,
  tool,
  onPoint,
  onBox,
  onChangePrompt,
  onChangeBox,
  onChoose,
  onObject,
  onSelectPrompts,
  selectedPrompts,
}: Props) {
  const [gesture, setGesture] = useState<{ start: Point; end: Point } | null>(
    null,
  );
  const [edit, setEdit] = useState<EditState | null>(null);
  const layoutRef = useRef({ width: 0, height: 0 });
  const gestureStart = useRef<Point | null>(null);
  const editRef = useRef<EditState | null>(null);

  if (!uri || !imageSize) {
    return (
      <View style={styles.empty}>
        {onChoose ? (
          <Text onPress={onChoose} style={styles.emptyAction}>
            Open Image
          </Text>
        ) : (
          <Text style={styles.emptyTitle}>Add a prompt to find matches</Text>
        )}
      </View>
    );
  }

  const clamp = (value: Point): Point => {
    const size = layoutRef.current;
    return {
      x: Math.max(0, Math.min(size.width, value.x)),
      y: Math.max(0, Math.min(size.height, value.y)),
    };
  };

  const toImage = (value: Point): PromptPoint => {
    const size = layoutRef.current;
    const point = clamp(value);
    return [
      Math.round((point.x / size.width) * imageSize.width),
      Math.round((point.y / size.height) * imageSize.height),
    ];
  };

  const boxStyle = (box: Box) => ({
    left: `${(box[0] / imageSize.width) * 100}%` as `${number}%`,
    top: `${(box[1] / imageSize.height) * 100}%` as `${number}%`,
    width: `${((box[2] - box[0]) / imageSize.width) * 100}%` as `${number}%`,
    height: `${((box[3] - box[1]) / imageSize.height) * 100}%` as `${number}%`,
  });

  const pointStyle = (point: PromptPoint) => ({
    left: `${(point[0] / imageSize.width) * 100}%` as `${number}%`,
    top: `${(point[1] / imageSize.height) * 100}%` as `${number}%`,
  });

  const gestureStyle = gesture
    ? {
        left: Math.min(gesture.start.x, gesture.end.x),
        top: Math.min(gesture.start.y, gesture.end.y),
        width: Math.abs(gesture.end.x - gesture.start.x),
        height: Math.abs(gesture.end.y - gesture.start.y),
      }
    : null;

  const onLayout = (event: LayoutChangeEvent) => {
    const { width, height } = event.nativeEvent.layout;
    layoutRef.current = { width, height };
  };

  const nativePoint = (event: GestureResponderEvent): Point => ({
    x: event.nativeEvent.locationX,
    y: event.nativeEvent.locationY,
  });

  const webPoint = (event: WebPointerEvent<HTMLDivElement>): Point => {
    const bounds = event.currentTarget.getBoundingClientRect();
    layoutRef.current = { width: bounds.width, height: bounds.height };
    return { x: event.clientX - bounds.left, y: event.clientY - bounds.top };
  };

  const imagePoint = (event: WebPointerEvent<HTMLDivElement>): PromptPoint => {
    const canvas = event.currentTarget.closest(
      '[aria-label="Prompt canvas"]',
    );
    if (!canvas) return [0, 0];
    const bounds = canvas.getBoundingClientRect();
    return [
      ((event.clientX - bounds.left) / bounds.width) * imageSize.width,
      ((event.clientY - bounds.top) / bounds.height) * imageSize.height,
    ];
  };

  const beginGesture = (point: Point) => {
    const value = clamp(point);
    gestureStart.current = value;
    if (tool !== 'point') setGesture({ start: value, end: value });
  };

  const moveGesture = (point: Point) => {
    if (!gestureStart.current) return;
    if (tool !== 'point') {
      setGesture({ start: gestureStart.current, end: clamp(point) });
    }
  };

  const finishGesture = (point: Point) => {
    const first = gestureStart.current;
    gestureStart.current = null;
    setGesture(null);
    if (!first) return;
    const last = clamp(point);
    const moved =
      Math.abs(last.x - first.x) >= 6 || Math.abs(last.y - first.y) >= 6;
    if (tool === 'point') {
      if (!moved) onPoint?.(toImage(last));
      return;
    }
    if (tool === 'box') {
      if (!moved) return;
      const start = toImage(first);
      const end = toImage(last);
      if (Math.abs(end[0] - start[0]) < 4 || Math.abs(end[1] - start[1]) < 4) {
        return;
      }
      onBox?.([
        Math.min(start[0], end[0]),
        Math.min(start[1], end[1]),
        Math.max(start[0], end[0]),
        Math.max(start[1], end[1]),
      ]);
      return;
    }
    if (!moved) {
      onSelectPrompts?.([]);
      return;
    }
    const x0 = Math.min(first.x, last.x);
    const y0 = Math.min(first.y, last.y);
    const x1 = Math.max(first.x, last.x);
    const y1 = Math.max(first.y, last.y);
    const size = layoutRef.current;
    onSelectPrompts?.(
      prompts.flatMap((item, index) => {
        const center =
          item.kind === 'point'
            ? item.point
            : ([(item.box[0] + item.box[2]) / 2, (item.box[1] + item.box[3]) / 2] as PromptPoint);
        const x = (center[0] / imageSize.width) * size.width;
        const y = (center[1] / imageSize.height) * size.height;
        return x >= x0 && x <= x1 && y >= y0 && y <= y1 ? [index] : [];
      }),
    );
  };

  const beginPointEdit = (
    event: WebPointerEvent<HTMLDivElement>,
    index: number,
  ) => {
    const item = prompts[index];
    if (item.kind !== 'point') return;
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    onSelectPrompts?.([index]);
    const original = [...item.point] as PromptPoint;
    const next: EditState = {
      kind: 'point',
      index,
      origin: imagePoint(event),
      original,
      point: original,
    };
    editRef.current = next;
    setEdit(next);
  };

  const beginBoxEdit = (
    event: WebPointerEvent<HTMLDivElement>,
    index: number,
    mode: BoxMode,
  ) => {
    const item = prompts[index];
    if (item.kind !== 'box') return;
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    onSelectPrompts?.([index]);
    const original = [...item.box] as Box;
    const next: EditState = {
      kind: 'box',
      index,
      mode,
      origin: imagePoint(event),
      original,
      box: original,
    };
    editRef.current = next;
    setEdit(next);
  };

  const moveEdit = (event: WebPointerEvent<HTMLDivElement>) => {
    const current = editRef.current;
    if (!current) return;
    event.preventDefault();
    const point = imagePoint(event);
    const dx = point[0] - current.origin[0];
    const dy = point[1] - current.origin[1];
    if (current.kind === 'point') {
      const next: EditState = {
        ...current,
        point: [
          Math.max(0, Math.min(imageSize.width - 1, current.original[0] + dx)),
          Math.max(0, Math.min(imageSize.height - 1, current.original[1] + dy)),
        ],
      };
      editRef.current = next;
      setEdit(next);
      return;
    }

    const original = current.original;
    const minSize = 4;
    let box: Box;
    if (current.mode === 'move') {
      const width = original[2] - original[0];
      const height = original[3] - original[1];
      const x0 = Math.max(0, Math.min(imageSize.width - width, original[0] + dx));
      const y0 = Math.max(0, Math.min(imageSize.height - height, original[1] + dy));
      box = [x0, y0, x0 + width, y0 + height];
    } else {
      let [x0, y0, x1, y1] = original;
      if (current.mode.includes('w')) x0 = Math.max(0, Math.min(x1 - minSize, x0 + dx));
      if (current.mode.includes('e')) x1 = Math.min(imageSize.width, Math.max(x0 + minSize, x1 + dx));
      if (current.mode.includes('n')) y0 = Math.max(0, Math.min(y1 - minSize, y0 + dy));
      if (current.mode.includes('s')) y1 = Math.min(imageSize.height, Math.max(y0 + minSize, y1 + dy));
      box = [x0, y0, x1, y1];
    }
    const next: EditState = { ...current, box };
    editRef.current = next;
    setEdit(next);
  };

  const finishEdit = (event: WebPointerEvent<HTMLDivElement>) => {
    const current = editRef.current;
    if (!current) return;
    event.preventDefault();
    event.stopPropagation();
    event.currentTarget.releasePointerCapture(event.pointerId);
    editRef.current = null;
    setEdit(null);
    if (current.kind === 'point') {
      const point = current.point.map(Math.round) as PromptPoint;
      if (point.some((value, index) => value !== current.original[index])) {
        onChangePrompt?.(current.index, point);
      }
    } else {
      const box = current.box.map(Math.round) as Box;
      if (box.some((value, index) => value !== current.original[index])) {
        onChangeBox?.(current.index, box);
      }
    }
  };

  const cancel = () => {
    gestureStart.current = null;
    editRef.current = null;
    setGesture(null);
    setEdit(null);
  };

  const gestureColor = tool === 'select' ? '#FFFFFF' : positive ? '#4DE0A7' : '#FF6B63';
  const cursor = tool === 'select' ? 'default' : 'crosshair';

  return (
    <View
      accessibilityLabel={label}
      onLayout={onLayout}
      onResponderGrant={(event) => beginGesture(nativePoint(event))}
      onResponderMove={(event) => moveGesture(nativePoint(event))}
      onResponderRelease={(event) => finishGesture(nativePoint(event))}
      onResponderTerminate={cancel}
      onStartShouldSetResponder={() =>
        Platform.OS !== 'web' && interactive && !disabled
      }
      style={[styles.canvas, { aspectRatio: imageSize.width / imageSize.height }]}
    >
      <Image
        source={{ uri }}
        resizeMode="stretch"
        style={{ width: '100%', height: '100%' }}
      />

      {Platform.OS === 'web' && interactive && !disabled && (
        <div
          aria-label={
            tool === 'select'
              ? 'Drag to select prompts'
              : tool === 'box'
                ? 'Drag to add a box prompt'
                : 'Click to add a point prompt'
          }
          onPointerCancel={cancel}
          onPointerDown={(event) => {
            event.preventDefault();
            event.currentTarget.setPointerCapture(event.pointerId);
            beginGesture(webPoint(event));
          }}
          onPointerMove={(event) => {
            if (!gestureStart.current) return;
            event.preventDefault();
            moveGesture(webPoint(event));
          }}
          onPointerUp={(event) => {
            if (!gestureStart.current) return;
            event.preventDefault();
            finishGesture(webPoint(event));
            event.currentTarget.releasePointerCapture(event.pointerId);
          }}
          role="application"
          style={{
            position: 'absolute',
            inset: 0,
            zIndex: 1,
            cursor,
            touchAction: 'none',
            userSelect: 'none',
          }}
        />
      )}

      <View style={[styles.fill, styles.passThrough]}>
        {objects.map((item) => (
          <View
            key={`object-${item.object_id}`}
            style={[styles.overlay, boxStyle(item.box)]}
          >
            <Image resizeMode="stretch" source={{ uri: item.mask }} style={styles.fill} />
            <View style={[styles.resultBox, { borderColor: item.color }]} />
          </View>
        ))}

        {prompts.map((item, index) => {
          const selected = selectedPrompts.includes(index);
          if (item.kind === 'point') {
            const point =
              edit?.kind === 'point' && edit.index === index
                ? edit.point
                : item.point;
            return (
              <View
                key={`prompt-${index}`}
                style={[
                  styles.point,
                  pointStyle(point),
                  {
                    backgroundColor: item.positive ? '#0A9D71' : '#D64C49',
                    borderColor: selected ? '#FFFFFF' : '#152033',
                  },
                ]}
              >
                <PromptSign positive={item.positive} />
              </View>
            );
          }
          const box =
            edit?.kind === 'box' && edit.index === index ? edit.box : item.box;
          return (
            <View
              key={`prompt-${index}`}
              style={[
                styles.promptBox,
                boxStyle(box),
                {
                  borderColor: selected
                    ? '#FFFFFF'
                    : item.positive
                      ? '#4DE0A7'
                      : '#FF6B63',
                  borderStyle: item.positive ? 'solid' : 'dashed',
                },
              ]}
            >
              <View
                style={[
                  styles.boxMark,
                  { backgroundColor: item.positive ? '#0A9D71' : '#D64C49' },
                ]}
              >
                <PromptSign positive={item.positive} />
              </View>
            </View>
          );
        })}

        {gestureStyle && (
          <View
            style={[
              styles.gesture,
              gestureStyle,
              {
                borderColor: gestureColor,
                backgroundColor:
                  tool === 'select'
                    ? 'rgba(100,217,194,0.18)'
                    : 'rgba(255,255,255,0.08)',
              },
            ]}
          />
        )}
      </View>

      {Platform.OS === 'web' && onObject && !disabled &&
        objects.map((item) => (
          <div
            aria-label={`Exclude result ${item.object_id}`}
            key={`object-action-${item.object_id}`}
            onClick={(event) => {
              event.preventDefault();
              event.stopPropagation();
              onObject(item);
            }}
            role="button"
            title="Exclude this result"
            style={{
              position: 'absolute',
              ...boxStyle(item.box),
              zIndex: 3,
              cursor: 'pointer',
            }}
          />
        ))}

      {Platform.OS === 'web' && interactive && !disabled && tool === 'select' &&
        prompts.map((item, index) => {
          if (item.kind === 'point') {
            const point =
              edit?.kind === 'point' && edit.index === index
                ? edit.point
                : item.point;
            return (
              <div
                aria-label={`${item.positive ? 'Include' : 'Exclude'} point ${index + 1}: select and move`}
                key={`editor-${index}`}
                onPointerCancel={cancel}
                onPointerDown={(event) => beginPointEdit(event, index)}
                onPointerMove={moveEdit}
                onPointerUp={finishEdit}
                role="button"
                style={{
                  position: 'absolute',
                  ...pointStyle(point),
                  width: 30,
                  height: 30,
                  zIndex: 3,
                  transform: 'translate(-50%, -50%)',
                  cursor: 'grab',
                  touchAction: 'none',
                }}
              />
            );
          }
          const box =
            edit?.kind === 'box' && edit.index === index ? edit.box : item.box;
          const selected = selectedPrompts.includes(index);
          return (
            <div
              aria-label={`${item.positive ? 'Include' : 'Exclude'} box ${index + 1}: select and move`}
              key={`editor-${index}`}
              onPointerCancel={cancel}
              onPointerDown={(event) => beginBoxEdit(event, index, 'move')}
              onPointerMove={moveEdit}
              onPointerUp={finishEdit}
              role="button"
              style={{
                position: 'absolute',
                ...boxStyle(box),
                zIndex: 3,
                cursor: 'move',
                touchAction: 'none',
              }}
            >
              {selected &&
                (['nw', 'ne', 'sw', 'se'] as BoxMode[]).map((mode) => (
                  <div
                    aria-label={`Resize ${mode}`}
                    key={mode}
                    onPointerCancel={cancel}
                    onPointerDown={(event) => beginBoxEdit(event, index, mode)}
                    onPointerMove={moveEdit}
                    onPointerUp={finishEdit}
                    role="slider"
                    style={{
                      position: 'absolute',
                      width: 12,
                      height: 12,
                      borderRadius: 6,
                      border: '2px solid #152033',
                      backgroundColor: '#FFFFFF',
                      left: mode.includes('w') ? -7 : undefined,
                      right: mode.includes('e') ? -7 : undefined,
                      top: mode.includes('n') ? -7 : undefined,
                      bottom: mode.includes('s') ? -7 : undefined,
                      cursor: `${mode}-resize`,
                      touchAction: 'none',
                    }}
                  />
                ))}
            </div>
          );
        })}

      {loadingText && (
        <View style={[styles.loading, styles.passThrough]}>
          <View style={styles.loadingCard}>
            <View style={styles.loadingDot} />
            <Text style={styles.loadingText}>{loadingText}</Text>
          </View>
        </View>
      )}
    </View>
  );
}

function PromptSign({ positive }: { positive: boolean }) {
  return (
    <View style={styles.promptSign}>
      <View style={styles.promptSignLine} />
      {positive && (
        <View style={[styles.promptSignLine, styles.promptSignVertical]} />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  empty: {
    minHeight: 520,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderStyle: 'dashed',
    borderColor: '#526078',
    borderRadius: 22,
    backgroundColor: '#111A29',
    padding: 32,
  },
  emptyTitle: { color: '#929CAA', fontSize: 14, fontWeight: '700' },
  emptyAction: {
    color: '#0A312D',
    backgroundColor: '#64D9C2',
    borderRadius: 12,
    paddingHorizontal: 18,
    paddingVertical: 11,
    fontSize: 14,
    fontWeight: '700',
    overflow: 'hidden',
  },
  canvas: {
    width: '100%',
    position: 'relative',
    overflow: 'hidden',
    borderRadius: 18,
    backgroundColor: '#0A101B',
  },
  fill: { position: 'absolute', inset: 0 },
  passThrough: { pointerEvents: 'none', zIndex: 2 },
  overlay: { position: 'absolute' },
  resultBox: {
    position: 'absolute',
    inset: 0,
    borderWidth: 1.5,
    borderRadius: 3,
  },
  point: {
    position: 'absolute',
    width: 22,
    height: 22,
    marginLeft: -11,
    marginTop: -11,
    borderRadius: 11,
    borderWidth: 3,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000000',
    shadowOpacity: 0.35,
    shadowRadius: 4,
  },
  promptSign: { width: 10, height: 10, alignItems: 'center', justifyContent: 'center' },
  promptSignLine: { position: 'absolute', width: 9, height: 2, borderRadius: 1, backgroundColor: '#FFFFFF' },
  promptSignVertical: { transform: [{ rotate: '90deg' }] },
  promptBox: { position: 'absolute', borderWidth: 2.5, borderRadius: 4 },
  boxMark: {
    position: 'absolute',
    top: -11,
    left: -11,
    width: 22,
    height: 22,
    borderRadius: 11,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 2,
    borderColor: '#152033',
  },
  gesture: { position: 'absolute', borderWidth: 1.5, borderStyle: 'dashed' },
  loading: {
    position: 'absolute',
    inset: 0,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(5,10,18,0.45)',
    zIndex: 4,
  },
  loadingCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: '#F6F2EA',
  },
  loadingDot: { width: 9, height: 9, borderRadius: 5, backgroundColor: '#0A9D71' },
  loadingText: { color: '#17202D', fontSize: 13, fontWeight: '700' },
});
