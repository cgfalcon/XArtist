"use client"

import Navbar from "../components/NavBar";
import ArtistCanvas from "@/app/components/ArtistCanvas";
import React from "react";

interface Props {
}

const AnonymousArtist = () => {
  // console.log(categories);
  return (
      <div className="container mx-auto">
          <ArtistCanvas/>
          {/* Add more sections as needed */}
      </div>
  );
};

export default AnonymousArtist;