
import Navbar from "../components/NavBar";
import React from "react";

interface Props {
}

const AnonymousArtist = () => {
  // console.log(categories);
  return (
      <div>
          <Navbar/>
          <div>
              <main>
                  <h1>The Anonymous Artist</h1>
              </main>
          </div>
          {/* Add more sections as needed */}
      </div>
  );
};

export default AnonymousArtist;