import React, { useState, useEffect } from 'react';

import CanvasBlock from "@/app/components/CanvasBlock";

function ArtistCanvas() {

    var canvasLayout = "Single" // "Grid" or "Single"

    if (canvasLayout == "Single") {
        return (
            <div className="grid place-items-center h-screen p-0">
                <CanvasBlock/>
            </div>
        );
    } else {
        return (
            <div className="grid grid-cols-4 gap-2 center">
                <CanvasBlock/>
                <CanvasBlock/>
                <CanvasBlock/>
                <CanvasBlock/>
                <CanvasBlock/>
                <CanvasBlock/>
                <CanvasBlock/>
                <CanvasBlock/>
            </div>


        );
    }


}

export default ArtistCanvas;