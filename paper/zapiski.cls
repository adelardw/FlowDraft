% Версия от 06.04.11
%

%
% Изменяем spmj-l.cls для нужд Записок
%
\NeedsTeXFormat{LaTeX2e}% LaTeX 2.09 can't be used (nor non-LaTeX)
[1994/12/01]% LaTeX date must December 1994 or later
\ProvidesClass{zapiski}
              [2011/04/06 v1.0 Zapiski nauchnykh seminarov POMI Author Class]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{amsart}}
\ProcessOptions

\LoadClass[reqno]{amsart}[2009/07/02]

\RequirePackage{textcomp}
% для рисования правильного значка параграфа

% Команда, которая устанавливает параматры для печати англоязычной статьи
\def\english{
\righthyphenmin=3
\def\prefacename{Preface}
\def\refname{References}
\def\abstractname{Abstract}
\def\bibname{Bibliography}
\def\chaptername{Chapter}
\def\appendixname{Appendix}
\def\contentsname{Contents}
\def\indexname{Index}
\def\figurename{Fig.}
\def\tablename{Table}
\def\partname{Part}
\def\pagename{Page}
\def\proofname{Proof}
\def\glossaryname{Glossary}
\@ifundefined{chapter}{}{\def\chaptername{Chapter}}
\def\keywordsname{Key words and phrases}

% Abstract в статье по-английски вставляется также как и в "Алгебре и анализе"
% Оттуда и позаимствовали команды
% Обнуляем две команды, чтобы не вставляли название по-английски в конце статьи
\def\engauthors##1{}
\def\engtitle##1{}

\global\def\@setabstractb{\relax}
\global\def\@setabstracta{%
  \ifvoid\abstractbox
  \else
    \skip@18\p@ \advance\skip@-\lastskip
    \advance\skip@-\baselineskip \vskip\skip@
    \box\abstractbox
    \prevdepth\z@ % because \abstractbox is a vtop
  \fi
}

\renewenvironment{abstract}{%
  \ifx\maketitle\relax
    \ClassWarning{\@classname}{Abstract should precede
      \protect\maketitle\space in AMS documentclasses; reported}%
  \fi
  \global\setbox\abstractbox=\vtop \bgroup
    \normalfont\Small
    \list{}{\labelwidth\z@
      \leftmargin3pc \rightmargin\leftmargin
      \listparindent\normalparindent \itemindent\z@
      \parsep\z@ \@plus\p@
      \let\fullwidthdisplay\relax
    }%
     \item\unskip\ignorespaces%
    \item[\hskip\labelsep\scshape\abstractname.]%
}{%
  \endlist\egroup
  \ifx\@setabstract\relax \@setabstracta \fi
}

} % Закрывающаяся скобка от английских определений

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Стилевые параметры русского языка

\frenchspacing
\righthyphenmin 2

% Размеры страницы. Уточнить!!!

\textwidth=113mm
\textheight=178mm 
\addtolength{\textheight}{-\topmargin}
\addtolength{\textheight}{-\headheight}
\addtolength{\textheight}{-\headsep}
\footskip=18pt
% 178mm было вместе с колонтитулом
%\calclayout % Что это?


% Так мы вставляем дату

\renewcommand{\datename}{Поступило}
%\def\@setdate{\flushright{\footnotesize\upshape\datename\enskip\@date}}
\def\@setdate{\vspace*{-\baselineskip}\vbox to -\baselineskip{\flushright{%\footnotesize
{\footnotesize\rmfamily\datename\enskip\@date}}}}

%
% Модифицируем куски amsart
%

\def\@setabstracta{\relax}
\let\@engauthors=\@empty
\let\@engtitle=\@empty
%
% Это нужно для перевода авторов и названия, а также запихивания abstract'а
% в конец статьи

\def\@setaddresses{\par
  \nobreak \begingroup
\footnotesize
  \def\author##1{\addvspace\bigskipamount}%
%  \def\\{\unskip, \ignorespaces}%
% У нас принято ручное форматирование
  \interlinepenalty\@M
  \def\address##1##2{\begingroup
%    \par\addvspace\bigskipamount\indent
    \par\addvspace\bigskipamount
% Отступ мы не делаем
    \@ifnotempty{##1}{(\ignorespaces##1\unskip) }%
%    {\scshape\ignorespaces##2}\par\endgroup}%
    {\noindent\ignorespaces\normalfont\rmfamily##2}\par\endgroup}%
   \def\curraddr##1##2{}
%  \def\curraddr##1##2{\begingroup
%    \@ifnotempty{##2}{\nobreak\indent{\itshape Current address}%
%      \@ifnotempty{##1}{, \ignorespaces##1\unskip}\/:\space
%      ##2\par}\endgroup}%
% А нужно ли нам такое? Уточнить!!!
  \def\email##1##2{\nobreak\addvspace\smallskipamount\begingroup
    \@ifnotempty{##2}{\nobreak\noindent{\itshape E-mail}\/:%
      \@ifnotempty{##1}{, \ignorespaces##1\unskip}\/\space
      \footnotesize\ttfamily##2\par}\endgroup}%
% У нас помельче и пробелов поменьше, отступ отсутствует
  \def\urladdr##1##2{\begingroup
    \@ifnotempty{##2}{\nobreak\indent{\itshape URL}%
      \@ifnotempty{##1}{, \ignorespaces##1\unskip}\/:\space
      \ttfamily##2\par}\endgroup}%
% URL пока не адаптировали. А надо ли? Как именно у нас их пишут? Уточнить!!!
  \addresses
  \endgroup
}

%
% %%%%%
%

% Док-во пишем жирным шрифтом

\renewenvironment{proof}[1][\proofname]{\par
  \pushQED{\qed}%
  \normalfont \topsep6\p@\@plus6\p@\relax
  \trivlist
  \item[\hskip\labelsep
        \bfseries
% было  \itshape
    #1\@addpunct{.}]\ignorespaces
}{%
  \popQED\endtrivlist\@endpefalse
}

%%%%%%%%%%

% Заводим команды \engtitle и \engauthors для написания перевода названия 
% и фамилий авторов, помещаемые в конце статьи

\newcommand*{\engtitle}[1]{\gdef\@engtitle{#1}}
\newcommand*{\engauthors}[1]{\gdef\@engauthors{#1}}

% Разбираемся с абстрактом
% В русском мы пишем его в конце статьи, безо всяких премудростей
% в английском сразу после заголовка, т.е. как в amsart'е

\renewenvironment{abstract}{%
  \global\setbox\abstractbox=\vtop \bgroup
    \normalfont
}{%
\egroup
  \ifx\@setabstract\relax \@setabstracta \fi
}

%
% Коллекционируем авторов глобально команду менять не пришлось, 
% поменяли только союз
%

\renewcommand{\andify}{%
%  \nxandlist{\unskip, }{\unskip{} и~}{\unskip{} и~}}
  \nxandlist{\unskip, }{\unskip, }{\unskip, }}
% по последним данным "и" не пишем

%
% А печатаем мы их жирным шрифтом и прижимаем влево
% Также как и заголовок

% Для этого заводим специальное окружение flushleftquote
% и переделываем обе выводящие команды

\newenvironment{flushleftquote}{%
  \list{}{\leftmargin12pt\raggedright}\item[]%
}{%
  \endlist
}

\def\@setauthors{%
  \begingroup
  \trivlist
  %\centering\footnotesize 
  \@topsep30\p@\relax
  \advance\@topsep by -\baselineskip
  \item\relax
  \andify\authors
  \def\\{\protect\linebreak}%
% \MakeUppercase{\authors}%
  {\begin{flushleftquote}\bfseries{\authors}\end{flushleftquote}}%
  \endtrivlist
  \vskip5pt
  \endgroup
}

\def\@settitle{\begin{flushleftquote}%
  \baselineskip14\p@\relax
    \bfseries
\uppercasenonmath\@title
  \@title
  \end{flushleftquote}%
}

% AMSовский копирайт не пишем

\def\@setcopyright{}

% рисуем черту под колонтитулом
% идеология с линейкой Львовского
% шрифт получается на пункт меньше и ничего с этим разумными средствами 
% пока не поделать, поэтому мы локально подставляем в scriptsize нужный 
% нам шрифт
 
\def\ch@ngefonts{\let\tempfont=\scriptsize \let\scriptsize=\footnotesize}
\def\Ch@ngefonts{\let\scriptsize=\tempfont}
\let\@@evenhead=\@evenhead
\renewcommand{\@evenhead}{\raisebox{0pt}[\headheight][0pt]%
{\ch@ngefonts\vbox{\hbox to\textwidth{\strut \@@evenhead}\Ch@ngefonts\hrule}}}
\let\@@oddhead=\@oddhead
\renewcommand{\@oddhead}{\raisebox{0pt}[\headheight][0pt]%
{\ch@ngefonts\vbox{\hbox to\textwidth{\strut \@@oddhead}\Ch@ngefonts\hrule}}}


% Определяем выпуск, том месяц и т.п. 
% Именно так для согласования с английской версией SPbMJ и Plenum'ом
\def\originfo#1#2#3#4{\def\origvolume{#1}\def\origissue{#2}%
  \def\origmonth{#3}\def\origyear{#4}}
\originfo{00}% volume number
  {0}%        % issue number
  {Yyyy}%     % month
  {0000}%     % year
\def\origpages#1#2{\def\origfirstpage{#1}\def\origlastpage{#2}}
\origpages{0}% first page
  {0}%        % last page
\def\enginfo#1#2#3#4{\def\engvolume{#1}\def\engissue{#2}%
  \def\engmonth{#3}\def\engyear{#4}}
\enginfo{00}% volume number
  {0}%        % issue number
  {Yyyy}%     % month
  {0000}%     % year

% Определяем команду для УДК
% Здесь не используется, только для согласования с Plenum'ом
\def\UDC#1{\def\@UDC{#1}}
\let\@UDC=\@empty
\def\@setUDC{\ifx\@empty\@UDC\else{\bfseries UDC \@UDC}\fi}


%    Определяем логотип
\def\publnamef{%
  \hbox{Записки научных}}
\def\publnames{%
  \hbox{семинаров ПОМИ}}
\def\russianvolinfo{%
  \hbox{Том \origvolume, \origyear\ г.}}
\AtBeginDocument{%
  \def\@serieslogo{%
    \vbox to\headheight{%
      \parindent\z@ \parfillskip\z@ \fontsize{8}{8\p@}\selectfont
      \noindent\hbox{\vbox{\hfill\publnamef\linebreak\indent\hfill\publnames
      \linebreak\indent\hfill\russianvolinfo}}\linebreak
      \null\hfill 
      \endgraf
      \vss
    }%
  }%
}

% Переделываем maketitle

\def\@maketitle{%
  \normalfont\normalsize
  \let\@makefnmark\relax  \let\@thefnmark\relax
%  \ifx\@empty\@subjclass\else \@footnotetext{\@setsubjclass}\fi %У нас такого нет
  \ifx\@empty\@keywords\else \@footnotetext{\@setkeywords}\fi
  \ifx\@empty\thankses\else \@footnotetext{%
    \def\par{\let\par\@par}\@setthanks}\fi
%%%%
  \@mkboth{\@nx{\shortauthors}}{\@nx\shorttitle}%
% почему-то эта команда не работает, вероятно она замещается чем-то из amsart
  \global\topskip24\p@\relax % 5.5pc   "   "   "     "     "
  \ifx\@empty\authors \else \@setauthors \fi
  \@settitle
  \@setabstracta
  \normalsize
% У нас посвящение после abstract'а
  \ifx\@empty\@dedicatory
  \else
%    \baselineskip18\p@
    \baselineskip14\p@
    \vtop{\flushright{\bfseries\@dedicatory\@@par}%
% прижимаем вправо, полужирным шрифтом
      \global\dimen@i\prevdepth}\prevdepth\dimen@i
  \fi
  \mycontentslabel
% вставляем вспомогательную строчку в aux-файл для удобства создания 
% оглавления
  \if@titlepage
    \newpage
  \else
    \dimen@18\p@ \advance\dimen@-\baselineskip
    \vskip\dimen@\relax
  \fi
} % end \@maketitle

% Запихиваем abstract в конец статьи

\def\@setabstractb{%
 \ifvoid\abstractbox
 \else
   \skip@8\p@ \advance\skip@-\lastskip
% Я бы сделал вертикальный отступ побольше
%   \advance\skip@-\baselineskip 
   \vskip\skip@
   \unvbox\abstractbox
   \prevdepth\z@ % because \abstractbox is a vtop
 \fi
}

% У нас после "ключевых слов" двоеточие

\def\@setkeywords{%
  {\itshape \keywordsname}\/:\enspace \@keywords\@addpunct.}

% Пошли всякие заголовки

% У нас large sf

\def\section{\@startsection{section}{1}%
  \z@{.7\linespacing\@plus\linespacing}{.5\linespacing}%
%  {\normalfont\scshape\centering}}
  {\normalfont\large\scshape\centering}}
\def\specialsection{\@startsection{section}{1}%
  \z@{\linespacing\@plus\linespacing}{.5\linespacing}%
%  {\normalfont\centering}}
  {\normalfont\large\scshape\centering}}

% Остальные оформим, когда узнаем как они выглядят

% В списке литературы
% Мы пока ограничимся заменой квадратных скобок на число с точкой, но может
% сделаем общепринятый вариант? -- спросить С.В.
%
\def\@biblabel#1{\@ifnotempty{#1}{#1.}}

%
% Здесь мы определяем то чем это все закончится
%
\def\enddoc@text{%
  \ifx\@empty\@engauthors \else
  \ifx\@empty\@engtitle \else\@setengtitle\fi\fi
  \ifx\@empty\@setabstractb \else\@setabstractb\fi
%  \ifx\@empty\@translators \else\@settranslators\fi
  \ifx\@empty\@date \else\@setdate\fi
  \ifx\@empty\addresses \else\@setaddresses\fi
}

% Взято из spmj-l

%    Miscellaneous information associated with translation.
%    [bnb, 2003/06/03]
\def\translnote#1{%
  \ifx\@empty\@translnotes \def\@translnotes{#1}%
  \else\g@addto@macro\@translnotes{\par\vspace{3\p@}#1}\fi}
\let\@translnotes=\@empty
\def\@settranslnotes{%
  \par\begingroup
    \addvspace{3\p@\@plus5\p@}%
    \flushright\normalfont\Small
    \@translnotes\par
  \endgroup}
\newcommand{\eo}[1]{\translnote{English original provided by #1}}
\newcommand{\rv}[1]{\translnote{Revised version provided by #1}}
\newcommand{\op}[1]{\translnote{Originally published in #1}}
\newcommand{\eb}[1]{\translnote{Edited by \uppercase{#1}}}

% У нас вообще такое возникает? !!!

\def\@setengtitle{%
  \par\begingroup
    \addvspace{12\p@\@plus5\p@}%
    \normalfont
    \@engauthors\ \@engtitle\par
  \endgroup}

% Дописали перевод авторов и заголовка в конец статьи

%  Various style differences between SPMJ and amsart.
%  [bnb, 2004/12/14]
%  Add section sign before section numbers; also in table of contents.
 \def\@checksectlevel{\ifnum\@toclevel=\@ne \S\fi}
 \def\@seccntformat#1{%
   \protect\textup{\protect\@secnumfont
     \protect\@checksectlevel
     \csname the#1\endcsname
     \protect\@secnumpunct
   }%
 }

 \renewcommand{\tocsection}[3]{%
   \indentlabel{\@ifnotempty{#2}{\ignorespaces#1 \S#2.\quad}}#3}
% Если не надо всюду ставить значки параграфов, то можно закомментировать
% и просто пользуемся amsart'овским
% Обсудить с С.В. 

%  Subsection numbers bold.
\def\@secnumfont{%
    \ifnum\@toclevel=\tw@ \bfseries \fi
}
%  Omit page numbers from in-article contents.
\renewcommand{\addcontentsline}[3]{%
  \addtocontents{#1}{\protect\contentsline{#2}{#3}{}}}

%
% полезные заимствования из russcorr
%

%\newcommand*{\tg}{\mathop{\operator@font tg}\nolimits}
%\newcommand*{\ctg}{\mathop{\operator@font ctg}\nolimits}
%\newcommand*{\cosec}{\mathop{\operator@font cosec}\nolimits}
%\newcommand*{\arctg}{\mathop{\operator@font arctg}\nolimits}
%\newcommand*{\arcctg}{\mathop{\operator@font arcctg}\nolimits}
%\newcommand*{\sh}{\mathop{\operator@font sh}\nolimits}
%\newcommand*{\ch}{\mathop{\operator@font ch}\nolimits}
%\newcommand*{\cth}{\mathop{\operator@font cth}\nolimits}
\@ifundefined{th}{}{%
 \PackageInfo{russcorr}{Command \string\th\space is being redefined}
}
\def\th{\mathop{\operator@font th}\nolimits}
%\let\eps\varepsilon
%\let\vphi\varphi

\DeclareSymbolFont{AMSa}{U}{msa}{m}{n}
\DeclareSymbolFont{AMSb}{U}{msb}{m}{n}
\DeclareMathSymbol{\varkappa}{\mathord}{AMSb}{"7B}
\DeclareMathSymbol{\leqslant}{\mathrel}{AMSa}{"36}
\DeclareMathSymbol{\geqslant}{\mathrel}{AMSa}{"3E}

\def\contentsname{Оглавление}
\def\figurename{Рис.}
\def\partname{Часть}
%\def\listfigurename{Список рисунков}
%\def\listtablename{Список таблиц}
\def\refname{Литература}
% Прямо здесь можно менять шрифт
\def\bibname{Литература}
%\def\indexname{Предметный указатель}
\def\tablename{Таблица}
%\def\abstractname{Аннотация}
\@ifundefined{chapter}{}{\def\chaptername{Глава}}
\def\proofname{Доказательство}
\def\appendixname{Приложение}
%\def\prefacename{Предисловие}
\def\keywordsname{Ключевые слова}


%\if@textyo
%\DeclareTextComposite{\"}{OT1}{е}{`\ё}
%\DeclareTextComposite{\"}{OT1}{Е}{`\Ё}
%\fi

%
% Подписи к картинкам
%

\def\@captionheadfont{\rmfamily}
\def\@captionfont{\normalfont}

\def\CONT#1#2#3#4{\relax}
\def\mycontentslabel{\@bsphack
  \protected@write\@auxout{}%
         {\string\CONT{\authors}{\@title}{\origissue}{\thepage}}%
  \@esphack}
\def\nocontents{\gdef\mycontentslabel{\relax}}

% Кавычки-елочки
\chardef\llq=128
\chardef\rrq=129

\endinput

%%%%%%%%%%%%%
%
% С указателем разберемся потом, если он кому-нибудь нужен
%

\def\ukazatel{%
\def\ips{20mm}
\def\ps{10mm}
\frenchspacing
\rightskip=\ips
\parindent=0pt
\parfillskip=-\ips
% Команда, рисующая полужирным шрифтом авторов, обычным название, дальше
% \dotfill, затем номер журнала и страницы
\gdef\CONT##1##2##3##4{\hangindent=8mm{\bfseries\ignorespaces##1\unskip}
 {\mdseries\quad\ignorespaces##2\unskip}
 \dotfill\hbox to \ips{\hfil\itshape\ignorespaces##3\unskip\upshape,
  \hbox to \ps{\hfil\ignorespaces##4\unskip}}\par}
% упрощенная версия для ссылки типа
% Иванов И. И., см. Иванов И. И., Петров П. П.
\gdef\SCONT##1##2##3##4##5{\hangindent=8mm{\bfseries\ignorespaces##1,\unskip}
 {\mdseries\quad\ignorespaces см.~##2\unskip}
 \dotfill\hbox to \ips{\hfil\itshape\ignorespaces##4\unskip\upshape,
  \hbox to \ps{\hfil\ignorespaces##5\unskip}}\par}
%
% Для ежегодного указателя все генерируется само, поэтому в начале 
% документа нужно много всего добавить
%
\AtBeginDocument{%
  \def\@serieslogo{%
    \vbox to\headheight{%
      \parindent\z@ \parfillskip\z@ \fontsize{8}{8\p@}\selectfont
      \noindent\publname\linebreak
      \russianvolinfo
      \null\hfill 
      \endgraf
      \vss
    }%
  }%
\nocontents
\title[Именной указатель]%
{Именной указатель журнала \\ \llq Алгебра и анализ\rrq \\ 
$\text{т.~\origvolume, \origyear~г.}$}
\maketitle
\vbox to-1pc{\rightline{\upshape\No\hbox to\ps{\hfil с.}}\vss}
}
}


