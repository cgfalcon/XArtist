import Image from "next/image";
import styles from "./page.module.css";


import Hero from "./components/Hero";
import Navbar from "./components/NavBar";
import React from "react";

interface Props {
}

const Home = () => {
  // console.log(categories);
  return (
      <div>
          <Navbar/>
          <Hero/>
          {/* Add more sections as needed */}
      </div>
  );
};

export default Home;
