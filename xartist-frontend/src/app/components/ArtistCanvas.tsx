import React, { useState, useEffect } from 'react';

import CanvasBlock from "@/app/components/CanvasBlock";

function ArtistCanvas() {

    var canvasLayout = "Single" // "Grid" or "Single"

    if (canvasLayout == "Single") {
        return (
            <div className="relative flex">
                <CanvasBlock/>
            </div>
        );
    } else {
        return (
            <div className="grid grid-cols-2 gap-4">
                <CanvasBlock/>
                <CanvasBlock/>
            </div>

        );
    }


}

export default ArtistCanvas;