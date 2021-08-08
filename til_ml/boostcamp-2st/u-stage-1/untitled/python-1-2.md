---
description: '210802'
---

# \(Python 1-2강\) 파이썬 개요

### Python의 시작

* 1991년 귀도 반 로섬이 발표
* 플랫폼 독립적
* 인터프리터 언어
* 객체 지향
* 동적 타이핑 언어
* 처음은 C언어로 구현되었음

### Python 특징

* 독립적인 플랫폼
  * 운영체제에 상관없는 프로그램이라는 뜻
* 컴파일러 vs 인터프리터
  * 컴파일러는 소스코드-컴파일-어셈블-CPU 의 과정을 거친다
    * 컴파일이 OS에 맞게 이루어진다. 어셈블은 해당 OS의 방식으로 기계어를 해석
  * 인터프리터는 소스코드-인터프리트-CPU 의 과정을 거친다

<table>
  <thead>
    <tr>
      <th style="text-align:center">&#xCEF4;&#xD30C;&#xC77C;&#xB7EC;</th>
      <th style="text-align:center"></th>
      <th style="text-align:center">&#xC778;&#xD130;&#xD504;&#xB9AC;&#xD130;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center">
        <p>&#xC18C;&#xC2A4;&#xCF54;&#xB4DC;&#xB97C; &#xAE30;&#xACC4;&#xC5B4;&#xB85C;
          &#xBA3C;&#xC800; &#xBC88;&#xC5ED; &#xD6C4;</p>
        <p>&#xD574;&#xB2F9; &#xD50C;&#xB7AB;&#xD3FC;&#xC5D0; &#xCD5C;&#xC801;&#xD654;&#xB418;&#xC5B4;
          &#xD504;&#xB85C;&#xADF8;&#xB7A8; &#xC2E4;&#xD589;</p>
      </td>
      <td style="text-align:center">&#xC791;&#xB3D9;&#xBC29;&#xC2DD;</td>
      <td style="text-align:center">
        <p>&#xBCC4;&#xB3C4;&#xC758; &#xBC88;&#xC5ED;&#xACFC;&#xC815; &#xC5C6;&#xC774;
          <br
          />&#xC18C;&#xC2A4;&#xCF54;&#xB4DC;&#xB97C; &#xC2E4;&#xD589;&#xC2DC;&#xC810;&#xC5D0;
          &#xD574;&#xC11D; &#xD6C4;</p>
        <p>&#xCEF4;&#xD4E8;&#xD130;&#xAC00; &#xCC98;&#xB9AC;&#xD560; &#xC218; &#xC788;&#xB3C4;&#xB85D;
          &#xD568;</p>
      </td>
    </tr>
    <tr>
      <td style="text-align:center">&#xC2E4;&#xD589;&#xC18D;&#xB3C4;&#xAC00; &#xBE60;&#xB984;</td>
      <td style="text-align:center">&#xC7A5;&#xC810;</td>
      <td style="text-align:center">&#xAC04;&#xB2E8;&#xD568;, &#xC801;&#xC740; &#xBA54;&#xBAA8;&#xB9AC;</td>
    </tr>
    <tr>
      <td style="text-align:center">&#xB9CE;&#xC740; &#xBA54;&#xBAA8;&#xB9AC;</td>
      <td style="text-align:center">&#xB2E8;&#xC810;</td>
      <td style="text-align:center">&#xC2E4;&#xD589;&#xC18D;&#xB3C4;&#xAC00; &#xB290;&#xB9BC;</td>
    </tr>
    <tr>
      <td style="text-align:center">C, Java, C++, C#</td>
      <td style="text-align:center">&#xC8FC;&#xC694;&#xC5B8;&#xC5B4;</td>
      <td style="text-align:center">&#xD30C;&#xC774;&#xC36C;</td>
    </tr>
  </tbody>
</table>

* 동적 타이핑 언어
  * 프로그램이 실행하는 시점에 프로그램이 사용해야할 데이터에 대한 타입을 결정
  * 객체 지향적 언어는 실행 순서가 아니라 모듈\(객체\) 중심으로 프로그램을 작성한다. 이 때 객체는 행동과 속성을 가지고 있음
* 문법이 이해하기 쉽고 간단하며 직관적이다.
* 다양한 라이브러리가 존재한다.

