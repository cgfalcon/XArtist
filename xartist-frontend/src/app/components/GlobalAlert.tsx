"use client";

import React from 'react';
import { useError } from '../context/ErrorContext';

const GlobalAlert = () => {
    const { error } = useError();

    if (!error) return null;

    return (
        <div className="fixed top-30 right-4 bg-red-500 text-white px-4 py-2 rounded-md shadow-lg z-50 items-center">
            {error}
        </div>
    );
};

export default GlobalAlert;