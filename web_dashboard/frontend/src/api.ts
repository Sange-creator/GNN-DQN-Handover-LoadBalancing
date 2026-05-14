import axios from 'axios';

const API_BASE = 'http://localhost:8000/api';

export const fetchTopology = async () => {
  const response = await axios.get(`${API_BASE}/topology`);
  return response.data;
};

export const fetchSimulation = async () => {
  const response = await axios.get(`${API_BASE}/simulation`);
  return response.data;
};

export const fetchComparison = async () => {
  const response = await axios.get(`${API_BASE}/comparison`);
  return response.data;
};
