
import Navbar from "../components/NavBar";
import React from "react";

interface Props {
}

const Explore = () => {
  // console.log(categories);
  return (
      <div>
          <Navbar/>
          <div>
              <main>
                  <h1>Explore Features</h1>
              </main>
          </div>
          {/* Add more sections as needed */}
      </div>
  );
};

export default Explore;