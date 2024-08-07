'use client';

import type {Metadata} from "next";
import {Inter} from "next/font/google";
import Head from "next/head";
import "./globals.css";
import Navbar from "@/app/components/NavBar";
import React from "react";

import { ErrorProvider } from './context/ErrorContext';
import GlobalAlert from './components/GlobalAlert';
import useToken from '../hooks/useToken';
import Loading from './components/Loading';

const inter = Inter({subsets: ["latin"]});
//
// export const metadata: Metadata = {
//     title: "Create Next App",
//     description: "Generated by create next app",
// };

export default function RootLayout({
                                       children,
                                   }: Readonly<{
    children: React.ReactNode;
}>) {
    const token = useToken();

    if (!token) {
        return (<html lang="en">
            <body>
            <ErrorProvider>
                <GlobalAlert />
                <Navbar/>
                <Loading/>
            </ErrorProvider>
            </body>
        </html>)
    }

    return (
        <html lang="en">
        <Head>
        <title>The Anonymous Artist</title>
        </Head>
        <body className={`${inter.className} body-container  bg-gray-900 text-white`}>
        <div
            className="absolute inset-x-0 -top-40 -z-10 transform-gpu overflow-hidden blur-3xl sm:-top-80"
            aria-hidden="true"
        >
            <div
                className="relative left-[calc(50%-11rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 rotate-[30deg] bg-gradient-to-tr from-[#ff80b5] to-[#9089fc] opacity-30 sm:left-[calc(50%-30rem)] sm:w-[72.1875rem]"
                style={{
                    clipPath:
                        'polygon(74.1% 44.1%, 100% 61.6%, 97.5% 26.9%, 85.5% 0.1%, 80.7% 2%, 72.5% 32.5%, 60.2% 62.4%, 52.4% 68.1%, 47.5% 58.3%, 45.2% 34.5%, 27.5% 76.7%, 0.1% 64.9%, 17.9% 100%, 27.6% 76.8%, 76.1% 97.7%, 74.1% 44.1%)',
                }}
            />
        </div>
        <Navbar/>
        <div className="">
            <ErrorProvider>
                    <GlobalAlert />
                    {children}
                </ErrorProvider>
            {/*{children}*/}
        </div>
        <div
            className="absolute inset-x-0 top-[calc(100%-13rem)] -z-10 transform-gpu overflow-hidden blur-3xl sm:top-[calc(100%-30rem)]"
            aria-hidden="true"
        >
            <div
                className="relative left-[calc(50%+3rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 bg-gradient-to-tr from-[#ff80b5] to-[#9089fc] opacity-30 sm:left-[calc(50%+36rem)] sm:w-[72.1875rem]"
                style={{
                    clipPath:
                        'polygon(74.1% 44.1%, 100% 61.6%, 97.5% 26.9%, 85.5% 0.1%, 80.7% 2%, 72.5% 32.5%, 60.2% 62.4%, 52.4% 68.1%, 47.5% 58.3%, 45.2% 34.5%, 27.5% 76.7%, 0.1% 64.9%, 17.9% 100%, 27.6% 76.8%, 76.1% 97.7%, 74.1% 44.1%)',
                }}
            />
        </div>
        </body>
        </html>
    );
}
