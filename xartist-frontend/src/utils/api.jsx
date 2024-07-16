// /utils/api.jsx
import axios from 'axios';
import Cookies from 'js-cookie';

const api = axios.create({
  baseURL: 'http://127.0.0.1:5000/api',
  withCredentials: true, // This is important for sending cookies with the request
});

api.interceptors.request.use(
  (config) => {
    const token = Cookies.get('token');
    if (token) {
        config.headers['Authorization'] = `${token}`;
        config.headers['Token'] = `${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

export default api;