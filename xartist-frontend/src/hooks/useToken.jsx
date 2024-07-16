// /hooks/useToken.jsx
'use client';

import { useEffect, useState } from 'react';
import axios from 'axios';
import Cookies from 'js-cookie';
import api from '../utils/api';
import { useError } from '../app/context/ErrorContext';

const useToken = () => {
  const [token, setToken] = useState(null);
  // const { showError } = useError();

  useEffect(() => {
    const fetchToken = async () => {
      try {
        const response = await api.get('http://127.0.0.1:5000/api/authorization/acquire_token', { withCredentials: true });
        if (response.data.success) {
          Cookies.set('token', response.data.data.token);
          setToken(response.data.data.token);
        } else {
          console.error('Failed to acquire token:', response.data.error_msg);
          // showError(response.data.error_msg);
        }
      } catch (error) {
        console.error('Failed to fetch token:', error);
        showError('Failed to fetch token');
      }
    };

    const existingToken = Cookies.get('token');
    if (!existingToken) {
      fetchToken();
    } else {
      setToken(existingToken);
    }
  }, []);

  return token;
};

export default useToken;