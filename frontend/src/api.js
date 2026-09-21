import axios from "axios";

const api = axios.create({ baseURL: "/api", timeout: 300000 });

export const setupProject = () => api.post("/setup").then((r) => r.data);
export const fetchRuns = () => api.get("/runs").then((r) => r.data);
export const createRun = (payload) => api.post("/runs", payload).then((r) => r.data);
export const fetchRun = (id) => api.get(`/runs/${id}`).then((r) => r.data);
export const fetchDiff = (id) => api.get(`/runs/${id}/diff`).then((r) => r.data);

export default api;
