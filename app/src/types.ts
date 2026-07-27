export type Box = [number, number, number, number];
export type PromptPoint = [number, number];
export type PromptTool = 'select' | 'point' | 'box';

export type ImageSize = {
  width: number;
  height: number;
};

export type PointPromptMark = {
  kind: 'point';
  point: PromptPoint;
  positive: boolean;
};

export type BoxPromptMark = {
  kind: 'box';
  box: Box;
  positive: boolean;
};

export type PromptMark = PointPromptMark | BoxPromptMark;

export type ResultObject = {
  object_id: number;
  box: Box;
  mask: string;
  color: string;
  metrics: {
    score: number;
    similarity: number;
    negative_similarity?: number;
    mask_score?: number;
    refine_score?: number;
    stability_score?: number;
  };
};

export type SessionResult = ImageSize & {
  session_id: string;
  prompt_count: number;
  objects: ResultObject[];
};

export type Health = {
  status: string;
  device: string;
  model_loaded: boolean;
};
