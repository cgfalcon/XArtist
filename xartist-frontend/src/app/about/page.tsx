"use client"

import Navbar from "../components/NavBar";
import React from "react";

interface Props {
}

const AboutPage = () => {
  // console.log(categories);
  return (
      <div>
          <Navbar/>
          <div>
              <main>
                  <h1>About</h1>
              </main>
          </div>
          {/* Add more sections as needed */}
      </div>
  );
};

export default AboutPage;