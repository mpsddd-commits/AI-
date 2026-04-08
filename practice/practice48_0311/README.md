
CREATE TABLE db_air.`출발회수지연율` (
    도시 VARCHAR(50),
    지연 FLOAT,
    년도 INT,
    횟수 INT
);

CREATE TABLE db_air.`출발지연횟수` AS
SELECT air.`도시`,
	COUNT(CASE WHEN t.`년도` = 1987 THEN 1 END) AS cnt_1987,
    ROUND(AVG(CASE WHEN t.`년도` = 1987 
            THEN CAST(t.`출발지연시간` AS SIGNED) END),1) AS delay_1987,
    COUNT(CASE WHEN t.`년도` = 1988 THEN 1 END) AS cnt_1988,
    ROUND(AVG(CASE WHEN t.`년도` = 1988 
            THEN CAST(t.`출발지연시간` AS SIGNED) END),1) AS delay_1988,
    COUNT(CASE WHEN t.`년도` = 1989 THEN 1 END) AS cnt_1989,
    ROUND(AVG(CASE WHEN t.`년도` = 1989 
            THEN CAST(t.`출발지연시간` AS SIGNED) END),1) AS delay_1989
FROM db_air.`비행` AS t
INNER JOIN db_air.`항공사` AS air
    ON t.`출발공항코드` = air.`항공사코드`
WHERE air.`도시` IN ('San Francisco','St Louis','New York','Los Angeles','Houston','Denver','Dallas-Fort Worth','Chicago','Atlanta','Newark','Phoenix')
AND t.`출발지연시간` <> 'NA'
GROUP BY air.`도시`;

CREATE TABLE db_air.`도착지연횟수` AS
SELECT air.`도시`,
	COUNT(CASE WHEN t.`년도` = 1987 THEN 1 END) AS cnt_1987,
    ROUND(AVG(CASE WHEN t.`년도` = 1987 
            THEN CAST(t.`도착지연시간` AS SIGNED) END),1) AS delay_1987,
    COUNT(CASE WHEN t.`년도` = 1988 THEN 1 END) AS cnt_1988,
    ROUND(AVG(CASE WHEN t.`년도` = 1988 
            THEN CAST(t.`도착지연시간` AS SIGNED) END),1) AS delay_1988,
    COUNT(CASE WHEN t.`년도` = 1989 THEN 1 END) AS cnt_1989,
    ROUND(AVG(CASE WHEN t.`년도` = 1989 
            THEN CAST(t.`도착지연시간` AS SIGNED) END),1) AS delay_1989
FROM db_air.`비행` AS t
INNER JOIN db_air.`항공사` AS air
    ON t.`도착지공항코드` = air.`항공사코드`
WHERE air.`도시` IN ('San Francisco','St Louis','New York','Los Angeles','Houston','Denver','Dallas-Fort Worth','Chicago','Atlanta','Newark','Phoenix')
AND t.`도착지연시간` <> 'NA'
GROUP BY air.`도시`;


-- 년도별 비행 취소 건 제외 가장 많이 비행 출발한 도시
select a.`년도`, air.`도시` , COUNT(`출발공항코드`) as 횟수
	from db_air.`비행` a
	inner join db_air.`항공사` as air
	on(a.`출발공항코드` = air.`항공사코드`)
	where a.`비행취소여부` = 0
	and a.`년도` = 1987
	group by air.`도시`
	order by 횟수 desc
;

-- 년도별 비행횟수 순위 top 10
INSERT INTO db_air.`departDelay` (도시, 지연, 년도, 횟수)
SELECT 도시, 지연, 년도, 횟수
FROM (
    SELECT 
        air.`도시`,
        AVG(a.`출발지연시간`) AS 지연,
        a.`년도`,
        COUNT(a.`출발공항코드`) AS 횟수,
        ROW_NUMBER() OVER (
            PARTITION BY a.`년도`
            ORDER BY COUNT(a.`출발공항코드`) DESC
        ) AS 순위
    FROM db_air.`비행` AS a
    INNER JOIN db_air.`항공사` AS air
    ON a.`출발공항코드` = air.`항공사코드`
    WHERE a.`비행취소여부` = 0
    GROUP BY a.`년도`, air.`도시`
) AS t
WHERE 순위 <= 10;