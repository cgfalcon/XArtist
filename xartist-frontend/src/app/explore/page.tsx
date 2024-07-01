"use client"

import Navbar from "../components/NavBar";
import React from "react";
import { Slider } from "@material-tailwind/react";

interface Props {
}



const Explore = () => {
  // console.log(categories);
  return (
      <div>
      <div className="bg-white py-24 sm:py-32">
        <div className="mx-auto grid max-w-7xl gap-x-10 gap-y-20 px-6 lg:px-8 xl:grid-cols-2 yl:grid-rows-2">
          <div className="max-w-1xl max-l-2yl">
            <div className="grid gap-x-8 gap-y-12 sm:grid-rows-2 sm:gap-x-16 xl:row-span-2">
              <img
                className="object-cover object-center xl:row-span-1"
                src="https://www.researchgate.net/profile/Sergei-Astapov/publication/261467229/figure/fig2/AS:667663588143121@1536194812400/Three-clusters-in-a-three-dimensional-feature-space-Features-represent-energy-ratios-and.png"
                alt="nature image"
                style={{ height: 512, width: 512 }}
              />
              <div className="flex w-96 flex-col gap-12">
                <Slider size="lg" color="red" defaultValue={50} />
                <Slider size="lg" color="blue" defaultValue={50} />
                <Slider size="lg" color="green" defaultValue={50} />
              </div>
            </div>
          </div>
            
          <div className="max-l-2xl">  
            <img
              className="object-cover object-center yl:row-span-2"
              src="https://images.unsplash.com/photo-1682407186023-12c70a4a35e0?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2832&q=80"
              alt="nature image"
              style={{ height: 512, width: 512 }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Explore;