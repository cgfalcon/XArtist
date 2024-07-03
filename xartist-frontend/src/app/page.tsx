"use client"

import Image from "next/image";
import styles from "./page.module.css";


import Hero from "./components/Hero";
import Navbar from "./components/NavBar";
import React from "react";
import HeroBlock from "@/app/components/HeroBlock";

interface Props {
}

const Home = () => {
  // console.log(categories);
  return (
      <div className="relative">
          {/*<Navbar/>*/}
          <div className="fixed inset-0 bg-opacity-20">
              <HeroBlock/>
          </div>
          <div className="relative z-10">
             <Hero/>
          </div>
          {/* Add more sections as needed */}
      </div>
  );
};

export default Home;
