"use client"

import Navbar from "../components/NavBar";
import ArtistCanvas from "@/app/components/ArtistCanvas";
import {useRouter} from 'next/router';
import React, { useState } from "react";
import { convertDotsToImage } from "@/app/api/route";

import {
    Card,
    Input,
    Checkbox,
    Button,
    Typography,
} from "@material-tailwind/react";

export default function coardinator() {
    const [firstDot, setFirstDot] = useState('');
    const [secondDot, setSecondDot] = useState('');
    const [thirdDot, setThirdDot] = useState('');
    const [imgData, setImgData] = useState<ImageData | null>(null);

    const handleSubmit = async (event) => {
        event.preventDefault(); // Prevent the default form submission behavior
        // Call the API function and pass the state values
        console.log("firstDot", firstDot)
        console.log("secondDot", secondDot)
        console.log("thirdDot", thirdDot)
        const imageResponse = await convertDotsToImage(firstDot, secondDot, thirdDot);
        if (imageResponse) {
            setImgData(imageResponse); // Assuming `imageResponse` is the base64 string
        }
    };


    return (
        <div>
            <Navbar/>
            <div className="relative">
                <Card color="transparent" shadow={false}>
                    <Typography variant="h4" color="blue-gray">
                        Input Coordinates
                    </Typography>
                    <form className="mt-8 mb-2 w-80 max-w-screen-lg sm:w-96" onSubmit={handleSubmit}>
                        <div className="mb-1 flex flex-col gap-6">
                            <Input
                                size="lg"
                                label="1st-D"
                                value={firstDot}
                                onChange={e => setFirstDot(e.target.value)}
                                className=" !border-t-blue-gray-200 focus:!border-t-gray-900"
                            />
                            <Input
                                size="lg"
                                label="2nd-D"
                                value={secondDot}
                                onChange={e => setSecondDot(e.target.value)}
                                className=" !border-t-blue-gray-200 focus:!border-t-gray-900"
                            />
                            <Input
                                size="lg"
                                label="3rd-D"
                                value={thirdDot}
                                onChange={e => setThirdDot(e.target.value)}
                                className=" !border-t-blue-gray-200 focus:!border-t-gray-900"
                            />
                        </div>
                        <Button className="mt-6" fullWidth type="submit">
                            Submit
                        </Button>
                    </form>
                </Card>
                {imgData && (
                    <img
                        src={`data:image/jpeg;base64,${imgData}`}
                        alt="Generated Art"
                        style={{ width: '500px', height: '500px' }}
                        className="object-cover object-center w-full h-256 max-w-full rounded-lg"
                    />
                )}
            </div>
        </div>

    );
}