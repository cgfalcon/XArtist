import React, {useState, useEffect} from 'react';
import { useError } from "../context/ErrorContext";
import api from '../../utils/api';

export const dynamic = 'force-dynamic' // defaults to auto
export async function fetchImages(modelKey) {
    try {
        const rest = await fetch('http://127.0.0.1:5000/api/dynamic_block/fetch_images', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Fetch-Token': '120203',
            },
            body: JSON.stringify({ model: modelKey })
        });
        const resp = await rest.json()
        console.log('Fetch success')
        return resp
    } catch (error) {
        console.error('Failed to fetch images:', error);
    }

}

// export async function fetchImages(modelKey) {
//     try {
//         const rest = await api.post('http://localhost:5000/api/dynamic_block/fetch_images', { model: modelKey }, {
//             // method: 'POST',
//             headers: {
//                 'Content-Type': 'application/json',
//             }
//         });
//         const resp = await rest.json()
//         console.log('Fetch success')
//         return resp
//     } catch (error) {
//         console.error('Failed to fetch images:', error);
//     }
//
// }

export async function convertDotsToImage(first_dot, second_dot, third_dot) {
    const payload = {
        firstDot: first_dot,
        secondDot: second_dot,
        thirdDot: third_dot
    };

    try {
        const rest = await fetch(`http://127.0.0.1:5000/api/explorer/fetch_dots_to_img?1st_dot=${first_dot}&2nd_dot=${second_dot}&3rd_dot=${third_dot}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Fetch-Token': '120203',
            },
        });
        const resp = await rest.json()
        console.log('Fetch success')
        return resp.data
    } catch (error) {
        console.error('Convert dots to image failed:', error);
    }
}