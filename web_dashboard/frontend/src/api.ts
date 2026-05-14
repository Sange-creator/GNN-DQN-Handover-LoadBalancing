import axios from 'axios';

const API_BASE = 'http://localhost:8000/api';

export const fetchTopology = async () => {
  const r = await axios.get(`${API_BASE}/topology`);
  return r.data;
};

export const fetchTraining = async () => {
  const r = await axios.get(`${API_BASE}/training`);
  return r.data;
};

export const fetchComparison = async (scenario: string = 'dense_urban') => {
  const r = await axios.get(`${API_BASE}/comparison/${scenario}`);
  return r.data;
};

export const fetchSimulation = async () => {
  const r = await axios.get(`${API_BASE}/simulation`);
  return r.data;
};

export const fetchScenarios = async () => {
  const r = await axios.get(`${API_BASE}/scenarios`);
  return r.data;
};

export const fetchModelInfo = async () => {
  const r = await axios.get(`${API_BASE}/model-info`);
  return r.data;
};
