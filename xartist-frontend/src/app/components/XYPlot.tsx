import React, { useRef, useEffect } from "react";
import PropTypes from "prop-types";
import * as d3 from "d3";

const XYPlot = ({
    data,
    width,
    height,
    margin = { top: 20, bottom: 30, left: 30, right: 20 },
    xAccessor,
    yAccessor,
    onHover = () => { }
}) => {
    const svgRef = useRef();

    useEffect(() => {
        console.log("Data received by XYPlot:", data);
        const svg = d3.select(svgRef.current).attr("border", 1);

        svg.selectAll('*').remove(); // Clear the existing chart

        // Add SVG border
        svg
            .append("rect")
            .attr("height", height + margin.top + margin.bottom)
            .attr("width", width + margin.left + margin.right)
            .style("stroke", "black")
            .style("fill", "none")
            .style("stroke-width", 1)
            .style("fill", "white");

        // Add margins
        const chartGroup = svg
            .append("g")
            .attr("transform", `translate(${margin.left}, ${margin.top})`);

        const yScale = d3.scaleLinear()
            .domain([0, 1])
            .range([height, 0]);

        const xScale = d3.scaleLinear()
            .domain([0, 1])
            // .domain(d3.extent(data, xAccessor))
            .range([0, width]);

        // Add circles for data
        chartGroup.selectAll("circle")
            .data(data)
            .enter()
            .append("circle")
            .attr("cx", d => xScale(xAccessor(d)))
            .attr("cy", d => yScale(yAccessor(d)))
            .attr("r", 2);

        // Add hover callback
        const hoverRect = chartGroup.append("rect")
            .attr("height", height)
            .attr("width", width)
            .style("fill", "transparent");

        hoverRect.on("mousemove", (event) => {
            const [x, y] = d3.pointer(event);
            onHover({ y: yScale.invert(y), x: xScale.invert(x) });
        });
    }, [data, height, margin, onHover, width, xAccessor, yAccessor]);

    return (
        <svg
            ref={svgRef}
            width={width + margin.left + margin.right}
            height={height + margin.top + margin.bottom}
        />
    );
};

XYPlot.propTypes = {
    data: PropTypes.array.isRequired,
    width: PropTypes.number.isRequired,
    height: PropTypes.number.isRequired,
    margin: PropTypes.shape({
        top: PropTypes.number.isRequired,
        bottom: PropTypes.number.isRequired,
        left: PropTypes.number.isRequired,
        right: PropTypes.number.isRequired
    }),
    onHover: PropTypes.func,
    xAccessor: PropTypes.func.isRequired,
    yAccessor: PropTypes.func.isRequired,
};

export default XYPlot;

// import React, { Component, useRef, useEffect } from "react";
// import PropTypes from "prop-types";

// import * as d3 from "d3";


// // const XYPlot = ({ dots, setIsHovering }) => {
// //     const svgRef = React.useRef();

// //     React.useEffect(() => {
// //         if (dots.length > 0) {
// //             const svg = d3.select(svgRef.current);
// //             svg.selectAll('*').remove();

// //             svg
// //                 .selectAll('circle')
// //                 .data(dots)
// //                 .enter()
// //                 .append('circle')
// //                 .attr('cx', d => d.x * 400)
// //                 .attr('cy', d => d.y * 400)
// //                 .attr('r', 1)
// //                 .attr('fill', 'black');
// //         }
// //     }, [dots, setIsHovering]);

// //     return (
// //         <svg ref={svgRef} style={styles.plot}></svg>
// //     );
// // };

// // const styles = {
// //     plot: {
// //         width: '100%',
// //         height: '100%',
// //     },
// // };

// // ***Original XYPlot V_0

// class XYPlot extends Component {
//     constructor(props) {
//         super(props);
//         this.renderChart = this.renderChart.bind(this);
//     }

//     useEffect(() => {
//         if (dots.length > 0) {
//             const svg = d3.select(svgRef.current);
//             svg.selectAll('*').remove();

//             svg
//                 .selectAll('circle')
//                 .data(dots)
//                 .enter()
//                 .append('circle')
//                 .attr('cx', d => d.x * 400)
//                 .attr('cy', d => d.y * 400)
//                 .attr('r', 2)
//                 .attr('fill', 'black')
//                 .on('mouseover', () => setIsHovering(true))
//                 .on('mouseout', () => setIsHovering(false));
//         }
//     }, [dots, setIsHovering]);

//     componentDidMount() {
//         this.renderChart();
//     }

//     renderChart() {
//         let svg = d3.select(this.node).attr("border", 1);
//         const {
//             width,
//             height,
//             margin,
//             data,
//             xAccessor,
//             yAccessor,
//             // colorAccessor,
//             onHover
//         } = this.props;

//         // Add SVG border
//         svg
//             .append("rect")
//             .attr("height", height + margin.top + margin.bottom)
//             .attr("width", width + margin.left + margin.right)
//             .style("stroke", "black")
//             .style("fill", "none")
//             .style("stroke-width", 1)
//             .style("fill", "white");

//         // Add margins
//         svg = svg
//             .append("g")
//             .attr("transform", `translate(${margin.left}, ${margin.top})`);

//         const yScale = d3
//             .scaleLinear()
//             .domain(d3.extent(data, yAccessor))
//             .range([height, 0]);

//         const xScale = d3
//             .scaleLinear()
//             .domain(d3.extent(data, xAccessor))
//             .range([0, width]);

//         const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

//         // Add circles for data
//         svg
//             .selectAll("circle")
//             .data(data)
//             .enter()
//             .append("circle")
//             .attr("cx", d => xScale(xAccessor(d)))
//             .attr("cy", d => yScale(yAccessor(d)))
//             .attr("r", 2);
//             // .style("fill", d => colorScale(colorAccessor(d)));

//         // Add hover callback
//         const hoverRect = svg
//             .append("rect")
//             .attr("height", height)
//             .attr("width", width)
//             .style("fill", "transparent");

//         hoverRect.on("mousemove", () => {
//             const [x, y] = d3.pointer(hoverRect.node());
//             onHover({ y: yScale.invert(y), x: xScale.invert(x) });
//         });
//     }

//     render() {
//         return (
//             <svg
//                 ref={node => {
//                     this.node = node;
//                 }}
//                 width={
//                     this.props.width + this.props.margin.left + this.props.margin.right
//                 }
//                 height={
//                     this.props.height + this.props.margin.top + this.props.margin.bottom
//                 }
//             />
//         );
//     }
// }

// XYPlot.propTypes = {
//     data: PropTypes.array.isRequired,
//     width: PropTypes.number.isRequired,
//     height: PropTypes.number.isRequired,
//     margin: PropTypes.shape({
//         top: PropTypes.number.isRequired,
//         bottom: PropTypes.number.isRequired,
//         left: PropTypes.number.isRequired,
//         right: PropTypes.number.isRequired
//     }),
//     onHover: PropTypes.func,
//     xAccessor: PropTypes.func.isRequired,
//     yAccessor: PropTypes.func.isRequired,
//     // colorAccessor: PropTypes.func.isRequired
// };

// XYPlot.defaultProps = {
//     margin: {
//         top: 20,
//         bottom: 30,
//         left: 30,
//         right: 20
//     },
//     onHover: () => { }
// };

// export default XYPlot;