"use client"

import Navbar from "../components/NavBar";
import React from "react";

interface Props {
}

const people = [
    {
        name: 'Gang Chu',
        role: 'System Architect & AI Engineer',
        imageUrl:
            'https://avatars.githubusercontent.com/u/1491617?v=4',
    },
    {
        name: 'Keyan Lin',
        role: 'AI Engineer & Frontend Developer',
        imageUrl:
            'https://avatars.githubusercontent.com/u/163773315?v=4',
    },
    {
        name: 'Kaiyue Lu',
        role: 'AI Engineer (Clustering & Dimension Reduction)',
        imageUrl:
            'https://avatars.githubusercontent.com/u/82226049?v=4',
    },
    {
        name: 'Zhiyang Jin',
        role: 'AI Engineer & Frontend Developer',
        imageUrl:
            'https://avatars.githubusercontent.com/u/52263903?u=4f2817a94e7771ba35f884a3d2e5d39195c61f31&v=4',
    },
    // More people...
]

const AboutPage = () => {
  // console.log(categories);
  return (
      <div>
          <Navbar/>
          <div>
              <main>
                  <h1>About</h1>
              </main>
              <div className="bg-white py-24 sm:py-32">
                  <div className="mx-auto grid max-w-7xl gap-x-8 gap-y-20 px-6 lg:px-8 xl:grid-cols-3">
                      <div className="max-w-2xl">
                          <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">Meet our team</h2>
                          <p className="mt-6 text-lg leading-8 text-gray-600">
                              The 'Anonymous Artist'
                              Team#4 of SYDE660 Spring 2024, System Design Engineering, University of Waterloo
                          </p>
                      </div>
                      <ul role="list" className="grid gap-x-8 gap-y-12 sm:grid-cols-2 sm:gap-y-16 xl:col-span-2">
                          {people.map((person) => (
                              <li key={person.name}>
                                  <div className="flex items-center gap-x-6">
                                      <img className="h-16 w-16 rounded-full" src={person.imageUrl} alt="" />
                                      <div>
                                          <h3 className="text-base font-semibold leading-7 tracking-tight text-gray-900">{person.name}</h3>
                                          <p className="text-sm font-semibold leading-6 text-indigo-600">{person.role}</p>
                                      </div>
                                  </div>
                              </li>
                          ))}
                      </ul>
                  </div>
              </div>
          </div>
          {/* Add more sections as needed */}
      </div>
  );
};

export default AboutPage;