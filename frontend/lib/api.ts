const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface GenerationResponse {
  text: string;
}

interface SimulationResponse {
  graph_stats: {
    nodes: number;
    edges: number;
    density: number;
    average_degree: number;
    agent_counts: Record<string, number>;
    bot_ratio: number;
    influencer_ratio: number;
    skeptic_ratio: number;
  };
  propagation_metrics: {
    reach: number;
    depth: number;
    velocity: number;
    velocity_by_step: number[];
    echo_chamber_density: number;
  };
}

interface PredictionResponse {
  prediction: number;
  confidence: number;
}

interface ExplanationResponse {
  top_features: string[];
  importance_scores: number[];
}

interface ExplanationDriver {
  label: string;
  explanation: string;
  score: number;
  direction: string;
}

interface StructuredExplanationResponse {
  summary: string;
  reasoning: string[];
  risk_level: 'low' | 'medium' | 'high';
  key_drivers: ExplanationDriver[];
  recommendation: string;
}

interface AnalyzeResponse {
  generated_text: string;
  metrics: {
    graph_stats: SimulationResponse['graph_stats'];
    propagation_metrics: SimulationResponse['propagation_metrics'];
    features: number[];
  };
  prediction: number;
  confidence: number;
  threshold_used: number;
  explanation: StructuredExplanationResponse;
}

export interface NewsItem {
  title: string;
  description: string;
  source: string;
  url: string;
}

async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
    ...options,
  })

  if (!response.ok) {
    let detail = `API error: ${response.status} ${response.statusText}`
    try {
      const payload = await response.json()
      detail = payload?.detail ?? detail
    } catch {
      // ignore non-JSON error bodies
    }
    throw new Error(detail)
  }

  return response.json()
}

export async function generateMisinformation(topic: string): Promise<string> {
  const data = await fetchAPI<GenerationResponse>('/generate/misinformation', {
    method: 'POST',
    body: JSON.stringify({ topic }),
  })
  return data.text
}

export async function generateCounter(topic: string): Promise<string> {
  const data = await fetchAPI<GenerationResponse>('/generate/counter', {
    method: 'POST',
    body: JSON.stringify({ topic }),
  })
  return data.text
}

export async function generateNeutral(topic: string): Promise<string> {
  const data = await fetchAPI<GenerationResponse>('/generate/neutral', {
    method: 'POST',
    body: JSON.stringify({ topic }),
  })
  return data.text
}

export interface SimulationOverrides {
  bot_ratio_override?: number;
  seed_nodes?: number;
  influencer_strength?: number;
}

export async function simulate(
  text: string,
  steps: number = 10,
  overrides?: SimulationOverrides
): Promise<SimulationResponse> {
  return fetchAPI<SimulationResponse>('/simulate', {
    method: 'POST',
    body: JSON.stringify({ text, steps, ...overrides }),
  })
}

export async function predict(features: number[]): Promise<PredictionResponse> {
  return fetchAPI<PredictionResponse>('/predict', {
    method: 'POST',
    body: JSON.stringify({ features }),
  })
}

export async function explain(features: number[]): Promise<ExplanationResponse> {
  return fetchAPI<ExplanationResponse>('/explain', {
    method: 'POST',
    body: JSON.stringify({ features }),
  })
}

export async function analyze(topic: string, steps: number = 10): Promise<AnalyzeResponse> {
  return fetchAPI<AnalyzeResponse>('/analyze', {
    method: 'POST',
    body: JSON.stringify({ topic, steps }),
  })
}

export async function analyzeNews(title: string, description: string, steps: number = 10): Promise<AnalyzeResponse> {
  return fetchAPI<AnalyzeResponse>('/analyze/news', {
    method: 'POST',
    body: JSON.stringify({ title, description, steps }),
  })
}

export async function fetchTrendingNews(): Promise<NewsItem[]> {
  return fetchAPI<NewsItem[]>('/news/trending')
}

export async function fetchNewsByTopic(topic: string): Promise<NewsItem[]> {
  return fetchAPI<NewsItem[]>(`/news/search?topic=${encodeURIComponent(topic)}`)
}
