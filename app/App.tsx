import { StatusBar } from 'expo-status-bar';
import { useEffect, useRef, useState } from 'react';
import {
  Image,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  useWindowDimensions,
  View,
} from 'react-native';

import {
  addBox,
  addPoint,
  createSession,
  deletePoints,
  excludeObject,
  refineResults,
  updateBox,
  updatePoint,
} from './src/api';
import { Canvas } from './src/Canvas';
import {
  Box,
  ImageSize,
  PromptMark,
  PromptPoint,
  PromptTool,
  ResultObject,
} from './src/types';

type Phase = 'idle' | 'uploading' | 'ready' | 'inferencing';

export default function App() {
  const { width } = useWindowDimensions();
  const stacked = width < 900;
  const fileInput = useRef<HTMLInputElement>(null);
  const previewUrl = useRef<string | null>(null);
  const [phase, setPhase] = useState<Phase>('idle');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [imageSize, setImageSize] = useState<ImageSize | null>(null);
  const [prompts, setPrompts] = useState<PromptMark[]>([]);
  const [objects, setObjects] = useState<ResultObject[]>([]);
  const [positive, setPositive] = useState(true);
  const [tool, setTool] = useState<PromptTool>('point');
  const [selectedPrompts, setSelectedPrompts] = useState<number[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    return () => {
      if (previewUrl.current) URL.revokeObjectURL(previewUrl.current);
    };
  }, []);

  const chooseImage = () => {
    if (Platform.OS === 'web') fileInput.current?.click();
  };

  const openFile = async (file: File) => {
    if (previewUrl.current) URL.revokeObjectURL(previewUrl.current);
    const uri = URL.createObjectURL(file);
    previewUrl.current = uri;
    setImageUri(uri);
    setImageSize(null);
    setSessionId(null);
    setPrompts([]);
    setObjects([]);
    setPositive(true);
    setTool('point');
    setSelectedPrompts([]);
    setError(null);
    setPhase('uploading');
    Image.getSize(uri, (imageWidth, imageHeight) => {
      setImageSize({ width: imageWidth, height: imageHeight });
    });

    try {
      const data = await createSession(file);
      setSessionId(data.session_id);
      setImageSize({ width: data.width, height: data.height });
      setPhase('ready');
    } catch (reason) {
      setError(message(reason));
      setPhase('idle');
    }
  };

  const submitPoint = async (point: PromptPoint) => {
    if (!sessionId || phase !== 'ready') return;
    setError(null);
    setPhase('inferencing');
    try {
      const data = await addPoint(sessionId, point, positive);
      setPrompts((items) => [...items, { kind: 'point', point, positive }]);
      setSelectedPrompts([]);
      setObjects(data.objects);
    } catch (reason) {
      setError(message(reason));
    } finally {
      setPhase('ready');
    }
  };

  const submitBox = async (box: Box) => {
    if (!sessionId || phase !== 'ready') return;
    setError(null);
    setPhase('inferencing');
    try {
      const data = await addBox(sessionId, box, positive);
      setPrompts((items) => [...items, { kind: 'box', box, positive }]);
      setSelectedPrompts([]);
      setObjects(data.objects);
    } catch (reason) {
      setError(message(reason));
    } finally {
      setPhase('ready');
    }
  };

  const editPrompt = async (index: number, point: PromptPoint) => {
    if (!sessionId || phase !== 'ready') return;
    setError(null);
    setPhase('inferencing');
    try {
      const data = await updatePoint(sessionId, index, point);
      setPrompts((items) =>
        items.map((item, itemIndex) =>
          itemIndex === index && item.kind === 'point'
            ? { ...item, point }
            : item,
        ),
      );
      setObjects(data.objects);
    } catch (reason) {
      setError(message(reason));
    } finally {
      setPhase('ready');
    }
  };

  const editBox = async (index: number, box: Box) => {
    if (!sessionId || phase !== 'ready') return;
    setError(null);
    setPhase('inferencing');
    try {
      const data = await updateBox(sessionId, index, box);
      setPrompts((items) =>
        items.map((item, itemIndex) =>
          itemIndex === index && item.kind === 'box' ? { ...item, box } : item,
        ),
      );
      setObjects(data.objects);
    } catch (reason) {
      setError(message(reason));
    } finally {
      setPhase('ready');
    }
  };

  const removeSelected = async () => {
    if (!sessionId || !selectedPrompts.length || phase !== 'ready') return;
    const indices = [...selectedPrompts];
    setError(null);
    setPhase('inferencing');
    try {
      const data = await deletePoints(sessionId, indices);
      setPrompts((items) =>
        items.filter((_, itemIndex) => !indices.includes(itemIndex)),
      );
      setObjects(data.objects);
      setSelectedPrompts([]);
      if (data.prompt_count === 0) setPositive(true);
    } catch (reason) {
      setError(message(reason));
    } finally {
      setPhase('ready');
    }
  };

  const excludeResult = async (item: ResultObject) => {
    if (!sessionId || phase !== 'ready') return;
    setError(null);
    setPhase('inferencing');
    try {
      const data = await excludeObject(sessionId, item.object_id);
      setPrompts((items) => [
        ...items,
        { kind: 'box', box: item.box, positive: false },
      ]);
      setSelectedPrompts([]);
      setObjects(data.objects);
      setPositive(false);
    } catch (reason) {
      setError(message(reason));
    } finally {
      setPhase('ready');
    }
  };

  const refine = async () => {
    if (!sessionId || !objects.length || phase !== 'ready') return;
    setError(null);
    setPhase('inferencing');
    try {
      const data = await refineResults(sessionId);
      setObjects(data.objects);
    } catch (reason) {
      setError(message(reason));
    } finally {
      setPhase('ready');
    }
  };

  useEffect(() => {
    if (Platform.OS !== 'web') return;
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (event.key !== 'Delete' || target?.tagName === 'INPUT') return;
      event.preventDefault();
      void removeSelected();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  });

  const busy = phase === 'uploading' || phase === 'inferencing';
  const loadingText =
    phase === 'uploading'
      ? 'Encoding image…'
      : phase === 'inferencing'
        ? 'Updating masks…'
        : null;

  return (
    <View style={styles.page}>
      <StatusBar style="dark" />
      {Platform.OS === 'web' && (
        <input
          ref={fileInput}
          accept="image/*"
          aria-label="Choose image file"
          onChange={(event) => {
            const file = event.currentTarget.files?.[0];
            event.currentTarget.value = '';
            if (file) void openFile(file);
          }}
          style={{ display: 'none' }}
          type="file"
        />
      )}

      <ScrollView contentContainerStyle={styles.scroll}>
        <View style={styles.shell}>
          <View style={styles.header}>
            <Text style={styles.title}>Visual Match</Text>
            {Platform.OS === 'web' && imageUri && (
              <Button
                disabled={busy}
                label={sessionId ? 'Replace Image' : 'Choose Another Image'}
                onPress={chooseImage}
              />
            )}
          </View>

          {sessionId && (
            <View style={[styles.toolbar, stacked && styles.toolbarStacked]}>
              <View style={styles.controlGroup}>
                <Text style={styles.controlLabel}>PROMPT</Text>
                <View style={styles.tools}>
                  <ToolButton
                    active={tool === 'point'}
                    label="Point"
                    onPress={() => setTool('point')}
                  />
                  <ToolButton
                    active={tool === 'box'}
                    label="Box"
                    onPress={() => setTool('box')}
                  />
                </View>
              </View>
              <View style={styles.controlGroup}>
                <Text style={styles.controlLabel}>LABEL</Text>
                <View style={styles.modes}>
                  <ModeButton
                    active={positive}
                    color="#0A9D71"
                    label="Include"
                    onPress={() => setPositive(true)}
                    sign="plus"
                  />
                  <ModeButton
                    active={!positive}
                    color="#D64C49"
                    disabled={!prompts.length}
                    label="Exclude"
                    onPress={() => setPositive(false)}
                    sign="minus"
                  />
                </View>
              </View>
              <View style={styles.tools}>
                <ToolButton
                  active={tool === 'select'}
                  label="Edit"
                  onPress={() => setTool('select')}
                />
              </View>
              {objects.length > 0 && (
                <Button
                  disabled={busy}
                  label="Refine"
                  onPress={() => void refine()}
                  primary
                />
              )}
              {selectedPrompts.length > 0 && (
                <Button
                  danger
                  disabled={busy}
                  label={`Delete (${selectedPrompts.length})`}
                  onPress={() => void removeSelected()}
                />
              )}
            </View>
          )}

          {error && <Text style={styles.error}>{error}</Text>}

          <View
            style={[
              styles.panels,
              stacked && styles.panelsStacked,
              !imageUri && styles.panelsStart,
            ]}
          >
            <View style={[styles.panel, !imageUri && styles.startPanel]}>
              <View style={styles.panelHeader}>
                <Text style={styles.panelTitle}>Image</Text>
              </View>
              <Canvas
                disabled={busy || !sessionId}
                imageSize={imageSize}
                label="Prompt canvas"
                loadingText={phase === 'uploading' ? loadingText : null}
                objects={[]}
                onBox={(box) => void submitBox(box)}
                onChangeBox={(index, box) => void editBox(index, box)}
                onChangePrompt={(index, point) => void editPrompt(index, point)}
                onChoose={chooseImage}
                onPoint={(point) => void submitPoint(point)}
                onSelectPrompts={setSelectedPrompts}
                positive={positive}
                prompts={prompts}
                selectedPrompts={selectedPrompts}
                tool={tool}
                uri={imageUri}
              />
            </View>

            {imageUri && <View style={styles.panel}>
              <View style={styles.panelHeader}>
                <Text style={styles.panelTitle}>Results</Text>
                {objects.length > 0 && (
                  <View style={styles.resultCount}>
                    <Text style={styles.resultCountText}>{objects.length}</Text>
                  </View>
                )}
              </View>
              <Canvas
                disabled={busy}
                imageSize={imageSize}
                interactive={false}
                label="Mask results"
                loadingText={loadingText}
                objects={objects}
                onObject={(item) => void excludeResult(item)}
                positive={positive}
                prompts={[]}
                selectedPrompts={[]}
                tool="select"
                uri={imageUri}
              />
            </View>}
          </View>
        </View>
      </ScrollView>
    </View>
  );
}

function Button({
  label,
  onPress,
  primary = false,
  danger = false,
  disabled = false,
}: {
  label: string;
  onPress: () => void;
  primary?: boolean;
  danger?: boolean;
  disabled?: boolean;
}) {
  return (
    <Pressable
      disabled={disabled}
      onPress={onPress}
      style={({ pressed }) => [
        styles.button,
        primary && styles.buttonPrimary,
        danger && styles.buttonDanger,
        disabled && styles.disabled,
        pressed && !disabled && styles.pressed,
      ]}
    >
      <Text
        style={[
          styles.buttonText,
          primary && styles.buttonTextPrimary,
          danger && styles.buttonTextDanger,
        ]}
      >
        {label}
      </Text>
    </Pressable>
  );
}

function ModeButton({
  active,
  color,
  disabled = false,
  label,
  onPress,
  sign,
}: {
  active: boolean;
  color: string;
  disabled?: boolean;
  label: string;
  onPress: () => void;
  sign: 'plus' | 'minus';
}) {
  return (
    <Pressable
      disabled={disabled}
      onPress={onPress}
      style={[
        styles.modeButton,
        active && { backgroundColor: color },
        disabled && styles.disabled,
      ]}
    >
      <View style={styles.modeSign}>
        <View
          style={[
            styles.modeSignLine,
            active && styles.modeSignLineActive,
          ]}
        />
        {sign === 'plus' && (
          <View
            style={[
              styles.modeSignLine,
              styles.modeSignVertical,
              active && styles.modeSignLineActive,
            ]}
          />
        )}
      </View>
      <Text style={[styles.modeText, active && styles.modeTextActive]}>{label}</Text>
    </Pressable>
  );
}

function ToolButton({
  active,
  label,
  onPress,
}: {
  active: boolean;
  label: string;
  onPress: () => void;
}) {
  return (
    <Pressable
      onPress={onPress}
      style={[styles.toolButton, active && styles.toolButtonActive]}
    >
      <Text style={[styles.toolText, active && styles.toolTextActive]}>{label}</Text>
    </Pressable>
  );
}

function message(value: unknown) {
  return value instanceof Error ? value.message : 'Something went wrong.';
}

const styles = StyleSheet.create({
  page: { flex: 1, backgroundColor: '#F2EFE8' },
  scroll: { flexGrow: 1, padding: 18 },
  shell: { width: '100%', maxWidth: 1800, alignSelf: 'center' },
  header: {
    minHeight: 42,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 16,
  },
  title: { color: '#152033', fontSize: 21, fontWeight: '800', letterSpacing: -0.5 },
  toolbar: {
    minHeight: 40,
    marginTop: 12,
    marginBottom: 12,
    alignSelf: 'flex-start',
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: 8,
  },
  toolbarStacked: { flexWrap: 'wrap' },
  controlGroup: { gap: 4 },
  controlLabel: {
    color: '#7A818C',
    fontSize: 9,
    fontWeight: '800',
    letterSpacing: 0.8,
    paddingLeft: 4,
  },
  tools: { flexDirection: 'row', padding: 3, borderRadius: 11, backgroundColor: '#E8E5DE' },
  toolButton: { height: 36, minWidth: 70, borderRadius: 8, paddingHorizontal: 10, alignItems: 'center', justifyContent: 'center' },
  toolButtonActive: { backgroundColor: '#152033' },
  toolText: { color: '#68717E', fontSize: 12, fontWeight: '800' },
  toolTextActive: { color: '#FFFFFF' },
  modes: { flexDirection: 'row', padding: 3, borderRadius: 11, backgroundColor: '#E8E5DE' },
  modeButton: { height: 36, minWidth: 112, borderRadius: 8, paddingHorizontal: 13, flexDirection: 'row', gap: 7, alignItems: 'center', justifyContent: 'center' },
  modeSign: { width: 12, height: 12, alignItems: 'center', justifyContent: 'center' },
  modeSignLine: { position: 'absolute', width: 10, height: 2, borderRadius: 1, backgroundColor: '#68717E' },
  modeSignLineActive: { backgroundColor: '#FFFFFF' },
  modeSignVertical: { transform: [{ rotate: '90deg' }] },
  modeText: { color: '#68717E', fontSize: 12, lineHeight: 16, fontWeight: '800' },
  modeTextActive: { color: '#FFFFFF' },
  button: {
    height: 40,
    paddingHorizontal: 15,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#D4CFC6',
    backgroundColor: '#FFFFFF',
    alignItems: 'center',
    justifyContent: 'center',
  },
  buttonPrimary: { backgroundColor: '#152033', borderColor: '#152033' },
  buttonDanger: { backgroundColor: '#FFF3F1', borderColor: '#E7B6B0' },
  buttonText: { color: '#4F5865', fontSize: 12, fontWeight: '800' },
  buttonTextPrimary: { color: '#FFFFFF' },
  buttonTextDanger: { color: '#B9413D' },
  disabled: { opacity: 0.35 },
  pressed: { opacity: 0.75 },
  error: { color: '#AD3F39', fontSize: 12, fontWeight: '700', marginBottom: 10 },
  panels: { flexDirection: 'row', alignItems: 'flex-start', gap: 12 },
  panelsStacked: { flexDirection: 'column' },
  panelsStart: { justifyContent: 'center' },
  panel: {
    flex: 1,
    width: '100%',
    minWidth: 0,
    borderRadius: 18,
    backgroundColor: '#152033',
    padding: 10,
  },
  startPanel: { maxWidth: 720, alignSelf: 'center' },
  panelHeader: {
    height: 46,
    paddingHorizontal: 5,
    paddingBottom: 9,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  panelTitle: { color: '#F7F3EC', fontSize: 16, fontWeight: '800' },
  resultCount: { minWidth: 34, height: 34, borderRadius: 10, backgroundColor: '#DDF5ED', alignItems: 'center', justifyContent: 'center' },
  resultCountText: { color: '#087253', fontSize: 14, fontWeight: '900' },
});
