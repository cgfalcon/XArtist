import React, {useState, useEffect} from 'react';

export const dynamic = 'force-dynamic' // defaults to auto
export async function fetchImages(request: Request) {
    try {
        const rest = await fetch('http://localhost:5002/api/images');
        const data = await rest.json()
    } catch (error) {
        console.error('Failed to fetch images:', error);
    }

}