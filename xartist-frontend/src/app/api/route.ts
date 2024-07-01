import React, {useState, useEffect} from 'react';

export const dynamic = 'force-dynamic' // defaults to auto
export async function fetchImages() {
    try {
        const rest = await fetch('http://127.0.0.1:5000/api/dynamic_block/fetch_images', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Fetch-Token': '120203',
            }});
        const resp = await rest.json()
        console.log('Fetch success')
        return resp.data
    } catch (error) {
        console.error('Failed to fetch images:', error);
    }

}